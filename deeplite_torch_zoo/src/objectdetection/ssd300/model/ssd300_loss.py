import torch
import torch.nn as nn
from torch.autograd import Variable

# from deeplite_torch_zoo import _C as C
from deeplite_torch_zoo.src.objectdetection.ssd300.utils.utils import (Encoder, dboxes300_coco)


class Loss(nn.Module):
    """
    Implements the loss as the sum of the followings:
    1. Confidence Loss: All labels, with hard negative mining
    2. Localization Loss: Only on positive labels
    Suppose input dboxes has the shape 8732x4
    """

    def __init__(self, dboxes):
        super(Loss, self).__init__()
        self.scale_xy = 1.0 / dboxes.scale_xy
        self.scale_wh = 1.0 / dboxes.scale_wh

        self.sl1_loss = nn.SmoothL1Loss(reduce=False)
        self.dboxes = nn.Parameter(
            dboxes(order="xywh").transpose(0, 1).unsqueeze(dim=0), requires_grad=False
        )
        # Two factor are from following links
        # http://jany.st/post/2017-11-05-single-shot-detector-ssd-from-scratch-in-tensorflow.html
        self.con_loss = nn.CrossEntropyLoss(reduce=False)

    def _loc_vec(self, loc):
        """
        Generate Location Vectors
        """
        gxy = (
            self.scale_xy
            * (loc[:, :2, :] - self.dboxes[:, :2, :])
            / self.dboxes[
                :,
                2:,
            ]
        )
        gwh = self.scale_wh * (loc[:, 2:, :] / self.dboxes[:, 2:, :]).log()
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def preprocess(self, targets, labels_length):
        _targets = []
        bbox_offsets = [torch.tensor([0])]
        for llen, target in zip(labels_length, targets):
            # Here we add one to account for background class at index 0.
            # When we apply the model for evaluation we subtract 1 from
            # all labels to map classes between 0 and num_classes - 1
            target[:, 4] = target[:, 4] + 1
            _targets.append(target[:llen, :])
            bbox_offsets.append(bbox_offsets[-1] + llen)
        return torch.cat(_targets, dim=0), torch.cat(bbox_offsets)

    def _normalize_bboxes(self, bboxes, dims):
        _w, _h = dims
        bboxes[:, 0] = bboxes[:, 0] / _w
        bboxes[:, 2] = bboxes[:, 2] / _w
        bboxes[:, 1] = bboxes[:, 1] / _h
        bboxes[:, 3] = bboxes[:, 3] / _h
        return bboxes

    def _prepare_targets(self, targets, labels_length, images_shape, device="cuda"):
        targets, bbox_offsets = self.preprocess(targets, labels_length)
        bboxes = targets[:, :4].to(device)  # bboxes.to(device)
        labels = targets[:, 4].to(device, dtype=torch.long)  # labels.to(device)
        bbox_offsets = bbox_offsets.to(device)
        bboxes = self._normalize_bboxes(bboxes, images_shape)
        return bboxes, labels, bbox_offsets

    def compute_loss(
        self, img_shape, targets, labels_length, ploc, plabel, device="cuda"
    ):
        N = img_shape[0]
        bboxes, labels, bbox_offsets = self._prepare_targets(
            targets, labels_length, img_shape[2:], device=device
        )

        dboxes = dboxes300_coco()
        encoder = Encoder(dboxes, device)
        if False:  # not use_cpu: # need to Compile the csrc files.
            bboxes, labels = C.box_encoder(
                N, bboxes, bbox_offsets, labels, encoder.dboxes.to(device), 0.5
            )
            # output is ([N*8732, 4], [N*8732], need [N, 8732, 4], [N, 8732] respectively
            M = bboxes.shape[0] // N
            bboxes = bboxes.view(N, M, 4)
            labels = labels.view(N, M)
        else:
            bboxes, labels = encoder.encode_batch(bboxes, labels, bbox_offsets)
            bboxes = bboxes.to(device)
            labels = labels.to(device)

        ploc, plabel = ploc.float(), plabel.float()

        trans_bbox = bboxes.transpose(1, 2).contiguous().to(device)

        gloc = Variable(trans_bbox, requires_grad=False)
        glabel = Variable(labels, requires_grad=False)

        return self.forward(ploc, plabel, gloc, glabel)

    def forward(self, ploc, plabel, gloc, glabel):
        """
        ploc, plabel: Nx4x8732, Nxlabel_numx8732
            predicted location and labels

        gloc, glabel: Nx4x8732, Nx8732
            ground truth location and labels
        """
        mask = glabel > 0
        pos_num = mask.sum(dim=1)

        vec_gd = self._loc_vec(gloc)

        # sum on four coordinates, and mask
        sl1 = self.sl1_loss(ploc, vec_gd).sum(dim=1)
        sl1 = (mask.float() * sl1).sum(dim=1)

        # hard negative mining
        con = self.con_loss(plabel, glabel)

        # postive mask will never selected
        con_neg = con.clone()
        con_neg[mask] = 0
        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_idx.sort(dim=1)

        # number of negative three times positive
        neg_num = torch.clamp(3 * pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_mask = con_rank < neg_num

        # print(con.shape, mask.shape, neg_mask.shape)
        closs = (con * (mask.float() + neg_mask.float())).sum(dim=1)

        # avoid no object detected
        total_loss = sl1 + closs
        num_mask = (pos_num > 0).float()
        pos_num = pos_num.float().clamp(min=1e-6)
        ret = (total_loss * num_mask / pos_num).mean(dim=0)
        return ret
