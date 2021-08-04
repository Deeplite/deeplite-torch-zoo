import sys

import numpy as np

sys.path.append("../utils")
import torch
import torch.nn as nn

import deeplite_torch_zoo.src.objectdetection.configs.hyp_config as hyp_cfg
from deeplite_torch_zoo.src.objectdetection.yolov3.utils import tools
from deeplite_torch_zoo.src.objectdetection.yolov3.utils.data_augment import LabelSmooth
from deeplite_torch_zoo.src.objectdetection.yolov3.utils.tools import iou_xywh_numpy


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.__gamma = gamma
        self.__alpha = alpha
        self.__loss = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, input, target):
        loss = self.__loss(input=input, target=target)
        loss *= self.__alpha * torch.pow(
            torch.abs(target - torch.sigmoid(input)), self.__gamma
        )

        return loss


class YoloV3Loss(nn.Module):
    def __init__(self, num_classes=20, device="cuda"):
        super(YoloV3Loss, self).__init__()
        self.__iou_threshold_loss = hyp_cfg.TRAIN["IOU_THRESHOLD_LOSS"]
        self._strides = np.array(hyp_cfg.MODEL["STRIDES"])
        self.__num_classes = num_classes
        self._anchors = np.array(hyp_cfg.MODEL["ANCHORS"], dtype=float)
        self._anchors_per_scale = hyp_cfg.MODEL["ANCHORS_PER_SCLAE"]
        self.rank = device

    def _mutate_rank(self, rank):
        self.rank = rank

    def forward(self, p, p_d, targets, labels_length, img_size):
        (
            label_sbbox,
            label_mbbox,
            label_lbbox,
            sbboxes,
            mbboxes,
            lbboxes,
        ) = self.make_targets_batch(targets.cpu(), labels_length.cpu(), img_size)

        return self._forward(
            p, p_d, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
        )

    def _forward(
        self, p, p_d, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
    ):
        """
        :param p: Predicted offset values for three detection layers.
                    The shape is [p0, p1, p2], ex. p0=[bs, grid, grid, anchors, tx+ty+tw+th+conf+cls_20]
        :param p_d: Decodeed predicted value. The size of value is for image size.
                    ex. p_d0=[bs, grid, grid, anchors, x+y+w+h+conf+cls_20]
        :param label_sbbox: Small detection layer's label. The size of value is for original image size.
                    shape is [bs, grid, grid, anchors, x+y+w+h+conf+mix+cls_20]
        :param label_mbbox: Same as label_sbbox.
        :param label_lbbox: Same as label_sbbox.
        :param sbboxes: Small detection layer bboxes.The size of value is for original image size.
                        shape is [bs, 150, x+y+w+h]
        :param mbboxes: Same as sbboxes.
        :param lbboxes: Same as sbboxes
        """
        strides = self._strides

        loss_s, loss_s_giou, loss_s_conf, loss_s_cls = self.__cal_loss_per_layer(
            p[0], p_d[0], label_sbbox, sbboxes, strides[0]
        )
        loss_m, loss_m_giou, loss_m_conf, loss_m_cls = self.__cal_loss_per_layer(
            p[1], p_d[1], label_mbbox, mbboxes, strides[1]
        )
        loss_l, loss_l_giou, loss_l_conf, loss_l_cls = self.__cal_loss_per_layer(
            p[2], p_d[2], label_lbbox, lbboxes, strides[2]
        )

        loss = loss_l + loss_m + loss_s
        loss_giou = loss_s_giou + loss_m_giou + loss_l_giou
        loss_conf = loss_s_conf + loss_m_conf + loss_l_conf
        loss_cls = loss_s_cls + loss_m_cls + loss_l_cls

        return loss, loss_giou, loss_conf, loss_cls

    def make_targets_batch(self, _batch, labels_length, img_size):
        b_label_sbbox = []
        b_label_mbbox = []
        b_label_lbbox = []
        b_sbboxes = []
        b_mbboxes = []
        b_lbboxes = []
        for label_length, sample in zip(labels_length, _batch):
            # assert label_length > 0
            (
                label_sbbox,
                label_mbbox,
                label_lbbox,
                sbboxes,
                mbboxes,
                lbboxes,
            ) = self.make_targets_sample(sample[:label_length], img_size)

            b_label_sbbox.append(torch.from_numpy(label_sbbox).to(self.rank).float())
            b_label_mbbox.append(torch.from_numpy(label_mbbox).to(self.rank).float())
            b_label_lbbox.append(torch.from_numpy(label_lbbox).to(self.rank).float())
            b_sbboxes.append(torch.from_numpy(sbboxes).to(self.rank).float())
            b_mbboxes.append(torch.from_numpy(mbboxes).to(self.rank).float())
            b_lbboxes.append(torch.from_numpy(lbboxes).to(self.rank).float())

        b_label_sbbox = torch.stack(b_label_sbbox)
        b_label_mbbox = torch.stack(b_label_mbbox)
        b_label_lbbox = torch.stack(b_label_lbbox)
        b_sbboxes = torch.stack(b_sbboxes)
        b_mbboxes = torch.stack(b_mbboxes)
        b_lbboxes = torch.stack(b_lbboxes)

        return (
            b_label_sbbox,
            b_label_mbbox,
            b_label_lbbox,
            b_sbboxes,
            b_mbboxes,
            b_lbboxes,
        )

    def make_targets_sample(self, targets, img_size):
        """
        Label assignment. For a single picture all GT box bboxes are assigned anchor.
        1、Select a bbox in order, convert its coordinates("xyxy") to "xywh"; and scale bbox'
           xywh by the strides.
        2、Calculate the iou between the each detection layer'anchors and the bbox in turn, and select the largest
            anchor to predict the bbox.If the ious of all detection layers are smaller than 0.3, select the largest
            of all detection layers' anchors to predict the bbox.

        Note :
        1、The same GT may be assigned to multiple anchors. And the anchors may be on the same or different layer.
        2、The total number of bboxes may be more than it is, because the same GT may be assigned to multiple layers
        of detection.

        """
        train_output_size = img_size / self._strides
        label = [
            np.zeros(
                (
                    int(train_output_size[i]),
                    int(train_output_size[i]),
                    self._anchors_per_scale,
                    6 + self.__num_classes,
                )
            )
            for i in range(3)
        ]
        for i in range(3):
            label[i][..., 5] = 1.0

        bboxes_xywh = [
            np.zeros((150, 4)) for _ in range(3)
        ]  # Darknet the max_num is 30
        bbox_count = np.zeros((3,))

        for bbox in targets:
            bbox_coor = bbox[:4]
            bbox_class_ind = int(bbox[4])
            bbox_mix = bbox[5]

            # onehot
            one_hot = np.zeros(self.__num_classes, dtype=np.float32)
            one_hot[bbox_class_ind] = 1.0
            one_hot_smooth = LabelSmooth()(one_hot, self.__num_classes)

            # convert "xyxy" to "xywh"
            bbox_xywh = np.concatenate(
                [(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]],
                axis=-1,
            )

            bbox_xywh_scaled = (
                1.0 * bbox_xywh[np.newaxis, :] / self._strides[:, np.newaxis]
            )

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self._anchors_per_scale, 4))
                anchors_xywh[:, 0:2] = (
                    np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                )  # 0.5 for compensation
                anchors_xywh[:, 2:4] = self._anchors[i]

                iou_scale = iou_xywh_numpy(
                    bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh
                )
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    # Bug : 当多个bbox对应同一个anchor时，默认将该anchor分配给最后一个bbox
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:6] = bbox_mix
                    label[i][yind, xind, iou_mask, 6:] = one_hot_smooth

                    bbox_ind = int(bbox_count[i] % 150)  # BUG : 150为一个先验值,内存消耗大
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self._anchors_per_scale)
                best_anchor = int(best_anchor_ind % self._anchors_per_scale)

                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(
                    np.int32
                )

                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:6] = bbox_mix
                label[best_detect][yind, xind, best_anchor, 6:] = one_hot_smooth

                bbox_ind = int(bbox_count[best_detect] % 150)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh

        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __cal_loss_per_layer(self, p, p_d, label, bboxes, stride):
        """
        (1)The loss of regression of boxes.
          GIOU loss is defined in  https://arxiv.org/abs/1902.09630.

        Note: The loss factor is 2-w*h/(img_size**2), which is used to influence the
             balance of the loss value at different scales.
        (2)The loss of confidence.
            Includes confidence loss values for foreground and background.

        Note: The backgroud loss is calculated when the maximum iou of the box predicted
              by the feature point and all GTs is less than the threshold.
        (3)The loss of classes。
            The category loss is BCE, which is the binary value of each class.

        :param stride: The scale of the feature map relative to the original image

        :return: The average loss(loss_giou, loss_conf, loss_cls) of all batches of this detection layer.
        """
        BCE = nn.BCEWithLogitsLoss(reduction="none")
        FOCAL = FocalLoss(gamma=2, alpha=1.0, reduction="none")

        batch_size, grid = p.shape[:2]
        img_size = stride * grid

        p_conf = p[..., 4:5]
        p_cls = p[..., 5:]

        p_d_xywh = p_d[..., :4]

        label_xywh = label[..., :4]
        label_obj_mask = label[..., 4:5]
        label_cls = label[..., 6:]
        label_mix = label[..., 5:6]
        # loss giou
        giou = tools.GIOU_xywh_torch(p_d_xywh, label_xywh).unsqueeze(-1)

        # The scaled weight of bbox is used to balance the impact of small objects and large objects on loss.
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[..., 2:3] * label_xywh[..., 3:4] / (
            img_size ** 2
        )
        loss_giou = label_obj_mask * bbox_loss_scale * (1.0 - giou) * label_mix

        # loss confidence
        iou = tools.iou_xywh_torch(
            p_d_xywh.unsqueeze(4), bboxes.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        )
        iou_max = iou.max(-1, keepdim=True)[0]
        label_noobj_mask = (1.0 - label_obj_mask) * (
            iou_max < self.__iou_threshold_loss
        ).float()

        loss_conf = (
            label_obj_mask * FOCAL(input=p_conf, target=label_obj_mask)
            + label_noobj_mask * FOCAL(input=p_conf, target=label_obj_mask)
        ) * label_mix

        # loss classes
        loss_cls = label_obj_mask * BCE(input=p_cls, target=label_cls) * label_mix

        loss_giou = (torch.sum(loss_giou)) / batch_size
        loss_conf = (torch.sum(loss_conf)) / batch_size
        loss_cls = (torch.sum(loss_cls)) / batch_size
        loss = loss_giou + loss_conf + loss_cls

        return loss, loss_giou, loss_conf, loss_cls


if __name__ == "__main__":
    from model.yolov3 import Yolov3

    net = Yolov3()

    p, p_d = net(torch.rand(3, 3, 416, 416))
    label_sbbox = torch.rand(3, 52, 52, 3, 26)
    label_mbbox = torch.rand(3, 26, 26, 3, 26)
    label_lbbox = torch.rand(3, 13, 13, 3, 26)
    sbboxes = torch.rand(3, 150, 4)
    mbboxes = torch.rand(3, 150, 4)
    lbboxes = torch.rand(3, 150, 4)

    loss, loss_xywh, loss_conf, loss_cls = YoloV3Loss(num_classes=80)(
        p, p_d, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
    )
    print(loss)
