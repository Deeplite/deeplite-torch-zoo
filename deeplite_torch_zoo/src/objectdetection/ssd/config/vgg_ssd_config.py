import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors

class VGG_CONFIG():
    def __init__(self):
        self.image_size = 300
        self.image_mean = np.array([123, 117, 104])  # RGB layout
        self.image_std = 1.0

        self.iou_threshold = 0.45
        self.center_variance = 0.1
        self.size_variance = 0.2

        self.specs = [
            SSDSpec(38, 8, SSDBoxSizes(30, 60), [2]),
            SSDSpec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
            SSDSpec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
            SSDSpec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
            SSDSpec(3, 100, SSDBoxSizes(213, 264), [2]),
            SSDSpec(1, 300, SSDBoxSizes(264, 315), [2])
        ]


        self.priors = generate_ssd_priors(self.specs, self.image_size)