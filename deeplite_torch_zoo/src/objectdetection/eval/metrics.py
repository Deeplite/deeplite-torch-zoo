from abc import ABC, abstractmethod

import numpy as np
import torch


def calc_iou_tensor(box1, box2):
    """Calculation of IoU based on two boxes tensor,
    Reference to https://github.com/kuangliu/pytorch-src
    input:
        box1 (N, 4)
        box2 (M, 4)
    output:
        IoU (N, M)
    """
    N = box1.size(0)
    M = box2.size(0)

    be1 = box1.unsqueeze(1).expand(-1, M, -1)
    be2 = box2.unsqueeze(0).expand(N, -1, -1)

    # Left Top & Right Bottom
    lt = torch.max(be1[:, :, :2], be2[:, :, :2])
    # mask1 = (be1[:,:, 0] < be2[:,:, 0]) ^ (be1[:,:, 1] < be2[:,:, 1])
    # mask1 = ~mask1
    rb = torch.min(be1[:, :, 2:], be2[:, :, 2:])
    # mask2 = (be1[:,:, 2] < be2[:,:, 2]) ^ (be1[:,:, 3] < be2[:,:, 3])
    # mask2 = ~mask2

    delta = rb - lt
    delta[delta < 0] = 0
    intersect = delta[:, :, 0] * delta[:, :, 1]
    # *mask1.float()*mask2.float()

    delta1 = be1[:, :, 2:] - be1[:, :, :2]
    area1 = delta1[:, :, 0] * delta1[:, :, 1]
    delta2 = be2[:, :, 2:] - be2[:, :, :2]
    area2 = delta2[:, :, 0] * delta2[:, :, 1]

    iou = intersect / (area1 + area2 - intersect)
    return iou


class ObjectDetectionMetrics(ABC):
    def __init__(self, data, num_classes, threshold=0.5, maxDet=100):
        self.data = data
        self.threshold = threshold
        self.num_classes = num_classes
        self.maxDet = maxDet
        self.evalImgs = []

    def evaluate_sample(self, sample):
        det_labels = np.array(sample["detections"]["labels"])
        det_bboxes = np.array(sample["detections"]["bboxes"])

        gt_labels = np.array(sample["gt"]["labels"])
        gt_bboxes = np.array(sample["gt"]["bboxes"])

        results = {}
        for _cls in range(self.num_classes):
            cls_detections = det_bboxes[_cls == det_labels]
            cls_gt = gt_bboxes[_cls == gt_labels]

            if len(cls_gt) == 0 and len(cls_detections) == 0:
                continue

            dtind = np.argsort(-cls_detections[:, 4], kind="mergesort")
            cls_detections = cls_detections[dtind[0 : self.maxDet]]

            _ious = calc_iou_tensor(
                torch.tensor(cls_detections[:, 0:4]), torch.tensor(cls_gt)
            )

            gt_visited = np.zeros(cls_gt.shape[0])
            dt_visited = np.zeros(cls_detections.shape[0])

            dt_mapping = {}
            gt_mapping = {}

            for dind, d in enumerate(cls_detections):
                # information about best match so far (m=-1 -> unmatched)
                iou = self.threshold
                matched_index = -1
                for gind, g in enumerate(cls_gt):
                    # if this gt already matched, and not a crowd, continue
                    if gt_visited[gind] > 0:
                        continue

                    # continue to next gt unless better match made
                    if _ious[dind, gind] < iou:
                        continue
                    # if match successful and best so far, store appropriately
                    iou = _ious[dind, gind]
                    matched_index = gind
                # if match made store id of match for both dt and gt
                if matched_index == -1:
                    continue

                dt_visited[dind] = 1
                gt_visited[matched_index] = 1

                dt_mapping[dind] = matched_index
                gt_mapping[matched_index] = dind

            results[_cls] = {
                "detections": cls_detections,
                "dt_mapping": dt_mapping,
                "dt_visited": dt_visited,
                "cls_gtruth": cls_gt,
                "gt_visited": gt_visited,
                "gt_mapping": gt_mapping,
            }

        return results

    def evaluate(self):
        for sample in self.data:
            self.evalImgs.append(self.evaluate_sample(sample))

    def get_cls_results(self, _cls):
        assert len(self.evalImgs) > 0, "make sure to call evaluate first"
        cls_results = []
        for r in self.evalImgs:
            if _cls in r:
                cls_results.append(r[_cls])
        return cls_results

    def calc_precision(self, tp, fp, fn):
        pr = tp / (tp + fp + np.spacing(1))  # add EPS to the denominator
        rc = tp / (tp + fn + np.spacing(1))
        f = (2 * pr * rc) / (pr + rc + np.spacing(1))
        return pr, rc, f

    @abstractmethod
    def accumlate(self):
        pass


class MAP(ObjectDetectionMetrics):
    """
    Implements mean average precision:
    1. Detections contain bboxes coordinates along with confidence and labels.
    2. Ground truth contains bounding boxes coordinates and labels ids
    """

    def __init__(self, data, num_classes, threshold=0.5, maxDet=100):
        super().__init__(
            data=data, num_classes=num_classes, threshold=threshold, maxDet=maxDet
        )

        self.recThrs = np.linspace(
            0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
        )

    def accumlate(self):
        R = len(self.recThrs)

        precision = -np.ones(
            (self.num_classes, R)
        )  # -1 for the precision of absent categories
        ap = np.zeros((self.num_classes))
        recall = -np.ones((self.num_classes))
        scores = -np.ones((self.num_classes, R))

        for _cls in range(self.num_classes):
            cls_results = self.get_cls_results(_cls)

            if len(cls_results) == 0:
                continue

            dtScores = np.concatenate(
                [cls_result["detections"][:, 4] for cls_result in cls_results]
            )

            inds = np.argsort(-dtScores, kind="mergesort")
            dtScoresSorted = dtScores[inds]

            dt_matched = np.concatenate(
                [cls_result["dt_visited"] for cls_result in cls_results]
            )[inds]
            gt_matched = np.concatenate(
                [cls_result["gt_visited"] for cls_result in cls_results]
            )

            npig = len(gt_matched)

            tp = np.cumsum(dt_matched).astype(dtype=np.float)
            fp = np.cumsum(1 - dt_matched).astype(dtype=np.float)

            pr, rc, _ = self.calc_precision(tp, fp, len(gt_matched) - tp)

            nd = len(tp)
            rc = tp / npig

            q = np.zeros((R,))
            ss = np.zeros((R,))

            if nd:
                recall[_cls] = rc[-1]
            else:
                recall[_cls] = 0

            # numpy is slow without cython optimization for accessing elements
            # use python array gets significant speed improvement
            pr = pr.tolist()
            q = q.tolist()

            for i in range(nd - 1, 0, -1):
                if pr[i] > pr[i - 1]:
                    pr[i - 1] = pr[i]

            inds = np.searchsorted(rc, self.recThrs, side="left")
            try:
                for ri, pi in enumerate(inds):
                    q[ri] = pr[pi]
                    ss[ri] = dtScoresSorted[pi]
            except:
                pass

            precision[_cls, :] = q
            scores[_cls, :] = ss
            q = np.array(q)
            ap[_cls] = np.mean(q[q > -1])

        return ap
