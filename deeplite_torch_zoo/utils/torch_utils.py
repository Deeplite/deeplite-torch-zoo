# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

import os
import random
import warnings

from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

import torch
import torchvision
import torch.nn as nn

from torch.hub import load_state_dict_from_url
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from ultralytics.yolo.utils import colorstr
from ultralytics.yolo.utils.checks import check_version
from ultralytics.yolo.utils.torch_utils import (fuse_conv_and_bn, model_info, scale_img, ModelEMA,
                                                select_device, torch_distributed_zero_first, one_cycle)
from ultralytics.yolo.utils.ops import Profile

import deeplite_torch_zoo
from deeplite_torch_zoo.utils import LOGGER, get_file_hash


LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

TORCHVISION_0_10 = check_version(torchvision.__version__, '0.10.0')
TORCH_1_9 = check_version(torch.__version__, '1.9.0')
TORCH_1_11 = check_version(torch.__version__, '1.11.0')
TORCH_1_12 = check_version(torch.__version__, '1.12.0')
TORCH_2_0 = check_version(torch.__version__, minimum='2.0')


def generate_checkpoint_name(
    model,
    test_dataloader,
    pth_filename,
    model_name,
    dataset_name,
    metric_key='acc',
    ndigits=4,
):
    ckpt_hash = get_file_hash(pth_filename)
    model.load_state_dict(torch.load(pth_filename), strict=True)
    eval_fn = deeplite_torch_zoo.get_eval_function(
        model_name=model_name, dataset_name=dataset_name
    )
    metric_val = eval_fn(model, test_dataloader, progressbar=True)[metric_key]
    if isinstance(metric_val, torch.Tensor):
        metric_val = metric_val.item()
    metric_str = str(metric_val).lstrip('0').replace('.', '')[:ndigits]
    checkpoint_name = f'{model_name}_{dataset_name}_{metric_str}_{ckpt_hash}.pt'
    return checkpoint_name


def load_pretrained_weights(model, checkpoint_url, device='cpu'):
    pretrained_dict = load_state_dict_from_url(
        checkpoint_url,
        progress=True,
        check_hash=True,
        map_location=device,
    )
    load_state_dict_partial(model, pretrained_dict)
    return model


def load_state_dict_partial(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict and v.size() == model_dict[k].size()
    }  # pylint: disable=E1135, E1136
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    LOGGER.info(f'Loaded {len(pretrained_dict)}/{len(model_dict)} modules')


@contextmanager
def switch_train_mode(model, is_training=False):
    is_original_mode_training = model.training
    model.train(is_training)
    try:
        yield
    finally:
        model.train(is_original_mode_training)


def initialize_weights(model):
    """Initialize model weights to random values."""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True


def init_seeds(seed=0, deterministic=False):
    """Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
    # torch.backends.cudnn.benchmark = True  # AutoBatch problem https://github.com/ultralytics/yolov5/issues/9287
    if deterministic:  # https://github.com/ultralytics/yolov5/pull/8213
        if TORCH_2_0:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            os.environ['PYTHONHASHSEED'] = str(seed)
        else:
            LOGGER.warning('WARNING ⚠️ Upgrade to torch>=2.0.0 for deterministic training.')


class GenericLogger:
    """
    General purpose logger for non-task specific logging
    Usage: from utils.loggers import GenericLogger; logger = GenericLogger(...)
    Arguments
        opt:             Run arguments
        console_logger:  Console logger
        include:         loggers to include
    """

    def __init__(self, opt, console_logger, include=('tb', 'wandb')):
        # init default loggers
        self.save_dir = Path(opt.save_dir)
        self.include = include
        self.console_logger = console_logger
        self.csv = self.save_dir / 'results.csv'  # CSV logger
        if 'tb' in self.include:
            prefix = colorstr('TensorBoard: ')
            self.console_logger.info(
                f"{prefix}Start with 'tensorboard --logdir {self.save_dir.parent}', view at http://localhost:6006/")
            self.tb = SummaryWriter(str(self.save_dir))

    def log_metrics(self, metrics, epoch):
        # Log metrics dictionary to all loggers
        if self.csv:
            keys, vals = list(metrics.keys()), list(metrics.values())
            n = len(metrics) + 1  # number of cols
            s = '' if self.csv.exists() else (('%23s,' * n % tuple(['epoch'] + keys)).rstrip(',') + '\n')  # header
            with open(self.csv, 'a') as f:
                f.write(s + ('%23.5g,' * n % tuple([epoch] + vals)).rstrip(',') + '\n')

        if self.tb:
            for k, v in metrics.items():
                self.tb.add_scalar(k, v, epoch)

    def log_images(self, files, name='Images', epoch=0):
        # Log images to all loggers
        files = [Path(f) for f in (files if isinstance(files, (tuple, list)) else [files])]  # to Path
        files = [f for f in files if f.exists()]  # filter by exists

        if self.tb:
            for f in files:
                self.tb.add_image(f.stem, cv2.imread(str(f))[..., ::-1], epoch, dataformats='HWC')

    def log_graph(self, model, imgsz=(640, 640)):
        # Log model graph to all loggers
        if self.tb:
            log_tensorboard_graph(self.tb, model, imgsz)

    def log_model(self, model_path, epoch=0, metadata={}):
        # Log model to all loggers
        pass

    def update_params(self, params):
        # Update the paramters logged
        pass


def log_tensorboard_graph(tb, model, imgsz=(640, 640)):
    # Log model graph to TensorBoard
    try:
        p = next(model.parameters())  # for device, type
        imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz  # expand
        im = torch.zeros((1, 3, *imgsz)).to(p.device).type_as(p)  # input image (WARNING: must be zeros, not empty)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress jit trace warning
            tb.add_graph(torch.jit.trace(de_parallel(model), im, strict=False), [])
    except Exception as e:
        LOGGER.warning(f'WARNING ⚠️ TensorBoard graph visualization failure {e}')


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    # Simple early stopper
    def __init__(self, patience=30):
        self.best_fitness = 0.0  # i.e. mAP
        self.best_epoch = 0
        self.patience = patience or float('inf')  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch, fitness):
        if fitness >= self.best_fitness:  # >= 0 to allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        if stop:
            LOGGER.info(f'EarlyStopping patience {self.patience} exceeded, stopping training.')
        return stop


def smart_DDP(model):
    # Model DDP creation with checks
    assert not check_version(torch.__version__, '1.12.0', pinned=True), \
        'torch==1.12.0 torchvision==0.13.0 DDP training is not supported due to a known issue. ' \
        'Please upgrade or downgrade torch to use DDP. See https://github.com/ultralytics/yolov5/issues/8395'
    if check_version(torch.__version__, '1.11.0'):
        return DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, static_graph=True)
    else:
        return DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)


def smart_optimizer(model, name='Adam', lr=0.001, momentum=0.9, decay=1e-5):
    # 3-param group optimizer: 0) weights with decay, 1) weights no decay, 2) biases no decay
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        for p_name, p in v.named_parameters(recurse=0):
            if p_name == 'bias':  # bias (no decay)
                g[2].append(p)
            elif p_name == 'weight' and isinstance(v, bn):  # weight (no decay)
                g[1].append(p)
            else:
                g[0].append(p)  # weight (with decay)

    if name == 'Adam':
        optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
    elif name == 'AdamW':
        optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == 'RMSProp':
        optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == 'SGD':
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f'Optimizer {name} not implemented.')

    optimizer.add_param_group({'params': g[0], 'weight_decay': decay})  # add g0 with weight_decay
    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}) with parameter groups "
                f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias")
    return optimizer


def smartCrossEntropyLoss(label_smoothing=0.0):
    # Returns nn.CrossEntropyLoss with label smoothing enabled for torch>=1.10.0
    if check_version(torch.__version__, '1.10.0'):
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if label_smoothing > 0:
        LOGGER.warning(f'WARNING ⚠️ label smoothing {label_smoothing} requires torch>=1.10.0')
    return nn.CrossEntropyLoss()
