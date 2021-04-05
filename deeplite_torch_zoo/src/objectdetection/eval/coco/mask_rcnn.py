import torch
from .coco_evaluator import COCOEvaluator


class RCNNCOCOEvaluator(COCOEvaluator):
    def __init__(self, model, dataset, gt=None, net="rcnn", predictor=None, img_size=418):
        super().__init__(
            model=model,
            dataset=dataset,
            net=net,
            img_size=img_size,
            gt=gt,
        )

    def _tensorize(self, img):
        return torch.tensor(img.transpose(2, 0, 1) / 255.0, dtype=torch.float, device=self.device)

    def process_image(self, img, img_id):
        img = self._tensorize(img)
        self.model.eval()
        res = self.model([img])[0]
        boxes = res['boxes']
        labels = res['labels']
        probs = res['scores']

        results = []
        for bbox, label, prob in zip(boxes, labels, probs):
            xmin, ymin, xmax, ymax = bbox
            results.append(
                [
                    img_id,
                    xmin,
                    ymin,
                    xmax - xmin,
                    ymax - ymin,
                    prob,
                    self.dataset.add_coco_empty_category(label) - 1,
                ]
            )
        return results
