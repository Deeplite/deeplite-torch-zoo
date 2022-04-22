import random
import numpy as np
from torch.utils.data import Dataset

from deeplite_torch_zoo.src.objectdetection.datasets.data_augment import \
    Mixup, RandomAffine, RandomCrop, RandomHorizontalFlip, \
    Resize, AugmentHSV, RandomVerticalFlip, random_perspective


class DLZooDataset(Dataset):
    def __init__(self, hyp_cfg, img_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hyp_cfg = hyp_cfg
        self._img_size = img_size

    def _augment(self, img, bboxes):
        img, bboxes = random_perspective(img, bboxes,
            degrees=self._hyp_cfg['degrees'],
            translate=self._hyp_cfg['translate'],
            scale=self._hyp_cfg['scale'],
            shear=self._hyp_cfg['shear'],
            perspective=self._hyp_cfg['perspective'])
        transforms = [
            RandomHorizontalFlip(p=self._hyp_cfg['fliplr']),
            RandomVerticalFlip(p=self._hyp_cfg['flipud']),
            AugmentHSV(hgain=self._hyp_cfg['hsv_h'],
                       sgain=self._hyp_cfg['hsv_s'],
                       vgain=self._hyp_cfg['hsv_v']),
        ]
        for transform in transforms:
            img, bboxes = transform(np.copy(img), np.copy(bboxes))
        return img, bboxes

    def _load_mixup(self, item, get_img_fn, num_images, p=0.5):
        img_org, bboxes_org, img_id = get_img_fn(item)
        img_org = img_org.transpose(2, 0, 1)  # HWC->CHW

        item_mix = random.randint(0, num_images - 1)
        img_mix, bboxes_mix, _ = get_img_fn(item_mix)
        img_mix = img_mix.transpose(2, 0, 1)

        img, bboxes = Mixup(p=p)(img_org, bboxes_org, img_mix, bboxes_mix)
        return img, bboxes, img_id

    def _load_mosaic(self, item, get_img_fn, num_images, resize_to_original_size=True):
        # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        bboxes4 = []
        s = self._img_size
        mosaic_border = [-s // 2, -s // 2]
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in mosaic_border)  # mosaic center x, y
        indices = [item] + random.choices(range(0, num_images), k=3)  # 3 additional image indices
        random.shuffle(indices)

        for i, index in enumerate(indices):
            # Load image
            img, bboxes, img_id = get_img_fn(index)
            h, w = img.shape[:2]

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114.0, dtype=np.float32) #114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            if bboxes.size:
                bboxes[:, 0] += padw
                bboxes[:, 2] += padw
                bboxes[:, 1] += padh
                bboxes[:, 3] += padh

            bboxes4.append(bboxes)

        # Concat/clip labels
        bboxes4 = np.concatenate(bboxes4, 0)
        bboxes4 = np.clip(bboxes4, 0, img4.shape[0])

        if resize_to_original_size:
            img4, bboxes4 = Resize((s, s), True, False)(
                np.copy(img4), np.copy(bboxes4)
            )

        img4 = img4.transpose(2, 0, 1)  # HWC->CHW
        return img4, bboxes4, img_id
