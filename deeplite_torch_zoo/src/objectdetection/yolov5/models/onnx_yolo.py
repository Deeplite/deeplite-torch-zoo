# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""

import numpy as np
import torch
import torch.nn as nn


class ONNXYOLO5(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', device=torch.device('cpu'), dnn=False, data=None, fp16=False, fuse=True):
        # Usage:
        #   ONNX Runtime:                   *.onnx

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        self.device = device
        self.fp16 = fp16 & True
        self.nhwc = False  # BHWC formats (vs torch BCWH)
        self.stride = 32  # default stride
        cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA
        import onnxruntime
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(w, providers=providers)
        self.output_names = [x.name for x in self.session.get_outputs()]
        meta = self.session.get_modelmeta().custom_metadata_map  # metadata
        if 'stride' in meta:
            stride, names = int(meta['stride']), eval(meta['names'])

    def forward(self, im, augment=False, visualize=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

        im = im.cpu().numpy()  # torch to numpy
        y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
        y = y[0]
        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y], None
        else:
            return self.from_numpy(y), None

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

