import numpy as np

from vision.utils.box_utils import SSDBoxSizes, SSDSpec, generate_ssd_priors

class MOBILENET_CONFIG():
    def __init__(self):
        self.image_size = 300
        self.image_mean = np.array([127, 127, 127])  # RGB layout
        self.image_std = 128.0
        self.iou_threshold = 0.45
        self.center_variance = 0.1
        self.size_variance = 0.2

        self.specs = [
            SSDSpec(19, 16, SSDBoxSizes(60, 105), [2, 3]),
            SSDSpec(10, 32, SSDBoxSizes(105, 150), [2, 3]),
            SSDSpec(5, 64, SSDBoxSizes(150, 195), [2, 3]),
            SSDSpec(3, 100, SSDBoxSizes(195, 240), [2, 3]),
            SSDSpec(2, 150, SSDBoxSizes(240, 285), [2, 3]),
            SSDSpec(1, 300, SSDBoxSizes(285, 330), [2, 3]),
        ]

        self.priors = generate_ssd_priors(self.specs, self.image_size)
