import numpy as np
import torch

from deeplite_torch_zoo.src.objectdetection.datasets.data_augment import Resize
from deeplite_torch_zoo.src.objectdetection.yolov5.utils.general import nms, post_process


class Evaluator(object):
    def __init__(self, model, img_size=448, conf_thresh=0.001, nms_thresh=0.5):
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.val_shape = img_size
        self.model = model
        self.device = next(model.parameters()).device

    def get_bbox(self, img, multi_test=False, flip_test=False):
        if multi_test:
            test_input_sizes = range(320, 640, 96)
            bboxes_list = []
            for test_input_size in test_input_sizes:
                valid_scale = (0, np.inf)
                bboxes_list.append(self.__predict(img, test_input_size, valid_scale))
                if flip_test:
                    bboxes_flip = self.__predict(
                        img[:, ::-1], test_input_size, valid_scale
                    )
                    bboxes_flip[:, [0, 2]] = img.shape[1] - bboxes_flip[:, [2, 0]]
                    bboxes_list.append(bboxes_flip)
            bboxes = np.row_stack(bboxes_list)
        else:
            bboxes = self.__predict(img, self.val_shape, (0, np.inf))

        bboxes = nms(bboxes, self.conf_thresh, self.nms_thresh)

        return bboxes

    def apply_model(self, imgs):
        with torch.no_grad():
            pred_bbox, _ = self.model(imgs)

        pred_bbox = post_process(pred_bbox)
        return pred_bbox.squeeze()

    def __predict(self, img, test_shape, valid_scale):
        org_img = np.copy(img)
        org_h, org_w, _ = org_img.shape

        img = self.__get_img_tensor(img, test_shape).to(self.device)
        self.model.eval()
        pred_bbox = self.apply_model(img)
        if len(pred_bbox) == 0:
            return np.zeros((0, 6))

        bboxes = self._scale_predictions(
            pred_bbox, test_shape, (org_h, org_w), valid_scale
        )
        return bboxes

    def __get_img_tensor(self, img, test_shape):
        img = Resize((test_shape, test_shape), correct_box=False)(img, None).transpose(
            2, 0, 1
        )
        return torch.from_numpy(img[np.newaxis, ...]).float()

    def _scale_predictions(
        self, pred_bbox, test_input_size, org_img_shape, valid_scale
    ):
        """
        The prediction frame is filtered to remove frames with unreasonable scales
        """
        pred_coor = pred_bbox[:, :4]
        scores = pred_bbox[:, 4]
        classes = pred_bbox[:, 5]

        # (1)
        # (xmin_org, xmax_org) = ((xmin, xmax) - dw) / resize_ratio
        # (ymin_org, ymax_org) = ((ymin, ymax) - dh) / resize_ratio
        # It should be noted that no matter what data augmentation method we use during training, it does not affect the transformation method here
        # Suppose we use conversion method A for the input test image, then the conversion method for bbox here is the reverse process of method A
        org_h, org_w = org_img_shape
        resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
        dw = (test_input_size - resize_ratio * org_w) / 2
        dh = (test_input_size - resize_ratio * org_h) / 2
        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

        # (2) Cut off the part of the predicted bbox beyond the original image
        pred_coor = np.concatenate(
            [
                np.maximum(pred_coor[:, :2], [0, 0]),
                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1]),
            ],
            axis=-1,
        )
        # (3) Set the coor of the invalid bbox to 0
        invalid_mask = np.logical_or(
            (pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3])
        )
        pred_coor[invalid_mask] = 0

        # (4) Remove bboxes that are not within the valid range
        bboxes_scale = np.sqrt(
            np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1)
        )
        scale_mask = np.logical_and(
            (valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1])
        )

        # (5) Remove the bbox whose score is lower than score_threshold
        score_mask = scores > self.conf_thresh

        mask = np.logical_and(scale_mask, score_mask)

        coors = pred_coor[mask]
        scores = scores[mask]
        classes = classes[mask]

        bboxes = np.concatenate(
            [coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1
        )

        return bboxes
