# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# The file is modified by Deeplite Inc. from the original implementation on Mar 21, 2023

import math

import torch
import torch.nn as nn

from deeplite_torch_zoo.src.object_detection.yolo.heads import (
    Detect, DetectV8, DetectX
)
from deeplite_torch_zoo.src.dnn_blocks.common import ConvBnAct as Conv
from deeplite_torch_zoo.src.dnn_blocks.common import DWConv
from deeplite_torch_zoo.src.dnn_blocks.yolov7.repvgg_blocks import RepConv
from deeplite_torch_zoo.src.dnn_blocks.mobileone.mobileone_blocks import MobileOneBlock

from deeplite_torch_zoo.utils import (
    LOGGER,
    fuse_conv_and_bn,
    model_info,
    scale_img,
)


class DetectionModel(nn.Module):
    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(
            x, profile, visualize
        )  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = (
                p[..., 0:1] / scale,
                p[..., 1:2] / scale,
                p[..., 2:4] / scale,
            )  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(
        self, cf=None
    ):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(
                8 / (640 / s) ** 2
            )  # obj (8 objects per 640 image)
            b.data[:, 5:] += (
                math.log(0.6 / (m.nc - 0.999999))
                if cf is None
                else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6)
                % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean())
            )

    def fuse(self, verbose=False):
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                if verbose:
                    LOGGER.info(f'Fusing BN into Conv for module {m}')
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
            if isinstance(m, RepConv) or isinstance(m, MobileOneBlock):
                if verbose:
                    LOGGER.info(f'Fusing RepVGG-style module {m}')
                m.fuse()
        self.info()
        self._is_fused = True
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    def is_fused(self):
        return self._is_fused

    def _init_head(self, ch):
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor(
                [s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))]
            )
            m.anchors /= m.stride.view(-1, 1, 1)

            self.stride = m.stride
            self._initialize_biases()  # only run once
        if isinstance(m, DetectX):
            m.inplace = self.inplace
            self.stride = torch.tensor(m.stride)
            m.initialize_biases()  # only run once
        if isinstance(m, DetectV8):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)
            m.stride = torch.tensor(
                [s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))]
            )  # forward
            self.stride = m.stride
            m.bias_init()  # only run once
