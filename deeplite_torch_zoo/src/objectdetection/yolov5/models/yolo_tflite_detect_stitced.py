# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""

import ast
import contextlib
import zipfile
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
from urllib.parse import urlparse
import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
from scipy.special import expit as sigmoid
import yaml

def yaml_load(file='data.yaml'):
    # Single-line safe yaml loading
    with open(file, errors='ignore') as f:
        return yaml.safe_load(f)
anchors_h = np.array([
    [[1.23242, 1.89746], [1.45117, 3.74414], [2.90625, 3.59180]],
    [[0.41968, 0.77734], [0.84277, 1.62207], [1.28711, 3.05664]],
    [[1.34473, 3.41016], [2.56445, 4.22266], [4.73828, 5.64062]],
])
anchors_l = np.array([
    [[0.75830, 1.14453], [0.91357, 2.38672], [1.65723, 2.09766]],
    [[0.27100, 0.52832], [0.49390, 0.90430], [0.74121, 1.80078]],
    [[0.82861, 1.96387], [1.50195, 2.25000], [2.67383, 3.15625]],
])
class StitchedYoloTflite(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', device=torch.device('cpu'), dnn=False, data=None, fp16=False, fuse=True):
        # Usage:
        #   TensorFlow Lite:                *.tflite
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        nhwc = True # BHWC formats (vs torch BCWH)
        stride = 32  # default stride
        cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA
        try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
            from tflite_runtime.interpreter import Interpreter, load_delegate
        except ImportError:
            import tensorflow as tf
            Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate,
        # TFLite
        print(f'Loading {w} for TensorFlow Lite inference...')
        interpreter = Interpreter(model_path=w)  # load TFLite model
        interpreter.allocate_tensors()  # allocate
        input_details = interpreter.get_input_details()  # inputs
        output_details = interpreter.get_output_details()  # outputs
        # load metadata
        with contextlib.suppress(zipfile.BadZipFile):
            with zipfile.ZipFile(w, "r") as model:
                meta_file = model.namelist()[0]
                meta = ast.literal_eval(model.read(meta_file).decode("utf-8"))
                stride, names = int(meta['stride']), meta['names']
        # class names
        if 'names' not in locals():
            names = yaml_load(data)['names'] if data else {i: f'class{i}' for i in range(999)}
        self.anchors = anchors_h if '224px' in w or '192px' in w or '160' in w else anchors_l
   
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)
        im = im.cpu().numpy()
        input = self.input_details[0]
        int8 = input['dtype'] == np.uint8  # is TFLite quantized uint8 model
        if int8:
            scale, zero_point = input['quantization']
            im = (im / scale + zero_point).astype(np.uint8)  # de-scale
        self.interpreter.set_tensor(input['index'], im)
        self.interpreter.invoke()
        y = []
        anchors = self.anchors


        stride = np.array([16, 8, 32])
        index = 0; nc = 1; no = nc + 5; na = 3;
        for output in self.output_details:
            x = self.interpreter.get_tensor(output['index'])
            if int8:
                scale, zero_point = output['quantization']
                x = (x.astype(np.float32) - zero_point) * scale  # re-scale
            bs, ny, nx, _ = x.shape
            x = x.reshape(bs, nx, ny, na, no)
            x = np.transpose(x, (0, 3, 1, 2, 4))
            x = sigmoid(x)
            xy, wh, conf, _ = np.split(x, (2, 2 + 2, 4 + 1 + nc), 4) # 1 + nc 1 is for conf
            grid, anchor_grid = self._make_grid(anchors[index], stride[index], na, ny, nx)
            xy = ((xy * 2 + grid) * stride[index]) / (w, h) # xy
            wh = ((wh * 2) ** 2 * anchor_grid) / (w, h)  # wh
            x = np.concatenate((xy, wh, conf), axis=4)
            index = index + 1

            y.append(x.reshape(bs, na * nx * ny, no))
        y = np.concatenate(y, axis=1)
        y[0][..., :4] *= [w, h, w, h]

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y], None
        else:
            return self.from_numpy(y), None

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    @staticmethod
    def _make_grid(anchors=[], stride=32, na=3, ny=0, nx=0):
        shape = (1, na, ny, nx, 2)
        y, x = np.arange(ny), np.arange(nx)
        yv, xv = np.meshgrid(y, x, indexing='ij')
        grid = np.broadcast_to(np.stack((xv, yv), axis=2), shape) - 0.5
        anchor_grid = np.broadcast_to((anchors * stride).reshape(1, 3, 1, 1, 2), shape)
        return grid, anchor_grid

