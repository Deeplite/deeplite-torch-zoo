# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

import os
import copy
import math
import time
import random
import warnings
import contextlib

from copy import deepcopy
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from torch.hub import load_state_dict_from_url
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

import deeplite_torch_zoo
from deeplite_torch_zoo.utils import LOGGER, get_file_hash, colorstr, check_version

LOCAL_RANK = int(
    os.getenv('LOCAL_RANK', "-1")
)  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', "-1"))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', "1"))

TORCHVISION_0_10 = check_version(torchvision.__version__, '0.10.0')
TORCH_1_9 = check_version(torch.__version__, '1.9.0')
TORCH_1_11 = check_version(torch.__version__, '1.11.0')
TORCH_1_12 = check_version(torch.__version__, '1.12.0')
TORCH_2_0 = check_version(torch.__version__, minimum='2.0')

NORMALIZATION_LAYERS = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.GroupNorm,
    nn.SyncBatchNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.LayerNorm,
    nn.LocalResponseNorm,
)

TRAINABLE_LAYERS = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.Linear,
    nn.Embedding,
    nn.EmbeddingBag,
)


def smart_inference_mode(torch_1_9=check_version(torch.__version__, '1.9.0')):
    # Applies torch.inference_mode() decorator if torch>=1.9.0 else torch.no_grad() decorator
    def decorate(fn):
        return (torch.inference_mode if torch_1_9 else torch.no_grad)()(fn)

    return decorate


def generate_zoo_checkpoint_name(
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
            LOGGER.warning(
                'WARNING ⚠️ Upgrade to torch>=2.0.0 for deterministic training.'
            )


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
                f"{prefix}Start with 'tensorboard --logdir {self.save_dir.parent}', view at http://localhost:6006/"
            )
            self.tb = SummaryWriter(str(self.save_dir))

    def log_metrics(self, metrics, epoch):
        # Log metrics dictionary to all loggers
        if self.csv:
            keys, vals = list(metrics.keys()), list(metrics.values())
            n = len(metrics) + 1  # number of cols
            s = (
                ''
                if self.csv.exists()
                else (('%23s,' * n % tuple(['epoch'] + keys)).rstrip(',') + '\n')
            )  # header
            with open(self.csv, 'a', encoding="utf-8") as f:
                f.write(s + ('%23.5g,' * n % tuple([epoch] + vals)).rstrip(',') + '\n')

        if self.tb:
            for k, v in metrics.items():
                self.tb.add_scalar(k, v, epoch)

    def log_images(self, files, name='Images', epoch=0):
        # Log images to all loggers
        files = [
            Path(f) for f in (files if isinstance(files, (tuple, list)) else [files])
        ]  # to Path
        files = [f for f in files if f.exists()]  # filter by exists

        if self.tb:
            for f in files:
                self.tb.add_image(
                    f.stem, cv2.imread(str(f))[..., ::-1], epoch, dataformats='HWC'
                )

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
        im = (
            torch.zeros((1, 3, *imgsz)).to(p.device).type_as(p)
        )  # input image (WARNING: must be zeros, not empty)
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
    return type(model) in (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )


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

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm


    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

class EarlyStopping:
    # Simple early stopper
    def __init__(self, patience=30):
        self.best_fitness = 0.0  # i.e. mAP
        self.best_epoch = 0
        self.patience = patience or float(
            'inf'
        )  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch, fitness):
        if (
            fitness >= self.best_fitness
        ):  # >= 0 to allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (
            self.patience - 1
        )  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        if stop:
            LOGGER.info(
                f'EarlyStopping patience {self.patience} exceeded, stopping training.'
            )
        return stop


def smart_DDP(model):
    # Model DDP creation with checks
    assert not check_version(torch.__version__, '1.12.0', pinned=True), (
        'torch==1.12.0 torchvision==0.13.0 DDP training is not supported due to a known issue. '
        'Please upgrade or downgrade torch to use DDP. See https://github.com/ultralytics/yolov5/issues/8395'
    )
    if check_version(torch.__version__, '1.11.0'):
        return DDP(
            model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, static_graph=True
        )
    return DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)


def smart_optimizer(model, name='Adam', lr=0.001, momentum=0.9, decay=1e-5):
    # 3-param group optimizer: 0) weights with decay, 1) weights no decay, 2) biases no decay
    g = [], [], []  # optimizer parameter groups
    bn = tuple(
        v for k, v in nn.__dict__.items() if 'Norm' in k
    )  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        for p_name, p in v.named_parameters(recurse=0):
            if p_name == 'bias':  # bias (no decay)
                g[2].append(p)
            elif p_name == 'weight' and isinstance(v, bn):  # weight (no decay)
                g[1].append(p)
            else:
                g[0].append(p)  # weight (with decay)

    if name == 'Adam':
        optimizer = torch.optim.Adam(
            g[2], lr=lr, betas=(momentum, 0.999)
        )  # adjust beta1 to momentum
    elif name == 'AdamW':
        optimizer = torch.optim.AdamW(
            g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0
        )
    elif name == 'RMSProp':
        optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == 'SGD':
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    elif name == 'SAM':
        base_optimizer = torch.optim.Adam
        optimizer = SAM(g[2], base_optimizer, lr=lr, betas=(momentum, 0.999))
        # base_optimizer = torch.optim.SGD
        # optimizer = SAM(g[2], base_optimizer, lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f'Optimizer {name} not implemented.')

    optimizer.add_param_group(
        {'params': g[0], 'weight_decay': decay}
    )  # add g0 with weight_decay
    optimizer.add_param_group(
        {'params': g[1], 'weight_decay': 0.0}
    )  # add g1 (BatchNorm2d weights)
    LOGGER.info(
        f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}) with parameter groups "
        f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias"
    )
    return optimizer


def smooth_crossentropy(pred, gold, label_smoothing=0.1):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=label_smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - label_smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)

def smartCrossEntropyLoss(label_smoothing=0.0):
    # Returns nn.CrossEntropyLoss with label smoothing enabled for torch>=1.10.0
    if check_version(torch.__version__, '1.10.0'):
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if label_smoothing > 0:
        LOGGER.warning(
            f'WARNING ⚠️ label smoothing {label_smoothing} requires torch>=1.10.0'
        )
    return nn.CrossEntropyLoss()


class no_jit_trace:
    def __enter__(self):
        self.state = torch._C._get_tracing_state()
        torch._C._set_tracing_state(None)

    def __exit__(self, *args):
        torch._C._set_tracing_state(self.state)
        self.state = None


def weight_gaussian_init(model: torch.nn.Module):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            else:
                continue

    return model


def reshape_elements(elements, shapes, device):
    def broadcast_val(elements, shapes):
        ret_grads = []
        for e, sh in zip(elements, shapes):
            ret_grads.append(
                torch.stack([torch.Tensor(sh).fill_(v) for v in e], dim=0).to(device)
            )
        return ret_grads

    if isinstance(elements[0]) == list:
        outer = []
        for e, sh in zip(elements, shapes):
            outer.append(broadcast_val(e, sh))
    else:
        return broadcast_val(elements, shapes)
    return outer


def get_layerwise_metric_values(model, metric_fn, target_layer_types=None):
    metric_array = []
    target_layer_types = target_layer_types or TRAINABLE_LAYERS
    for layer in model.modules():
        if isinstance(layer, target_layer_types):
            metric_array.append(metric_fn(layer))
    return metric_array


def strip_optimizer(f='best.pt', s=''):
    # Strip optimizer from 'f' to finalize training, optionally save as 's'
    x = torch.load(f, map_location=torch.device('cpu'))
    if x.get('ema'):
        x['model'] = x['ema']  # replace model with ema
    for k in 'optimizer', 'training_results', 'wandb_id', 'ema', 'updates':  # keys
        x[k] = None
    x['epoch'] = -1
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1e6  # filesize
    LOGGER.info(
        f"Optimizer stripped from {f},{(' saved as %s,' % s) if s else ''} {mb:.1f}MB"
    )


def fuse_conv_and_bn(conv, bn):
    """Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/."""
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = (
        torch.zeros(conv.weight.size(0), device=conv.weight.device)
        if conv.bias is None
        else conv.bias
    )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
        torch.sqrt(bn.running_var + bn.eps)
    )
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    # Scales img(bs,3,y,x) by ratio constrained to gs-multiple
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))  # new size
    img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
    if not same_shape:  # pad/crop img
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


def copy_attr(a, b, include=(), exclude=()):
    """Copies attributes from object 'b' to object 'a', with options to include/exclude certain attributes."""
    for k, v in b.__dict__.items():
        inc_len = len(include)
        if inc_len and k not in include or k.startswith('_') or k in exclude:
            continue
        setattr(a, k, v)


class ModelEMA:
    """Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    To disable EMA set the `enabled` attribute to `False`.
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        """Create EMA."""
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (
            1 - math.exp(-x / tau)
        )  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.enabled = True

    def update(self, model):
        """Update EMA parameters."""
        if self.enabled:
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:  # true for FP16 and FP32
                    v *= d
                    v += (1 - d) * msd[k].detach()
                    # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype},  model {msd[k].dtype}'

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        """Updates attributes and saves stripped model with optimizer removed."""
        if self.enabled:
            copy_attr(self.ema, model, include, exclude)


def select_device(device='', batch=0, newline=False, verbose=True):
    """Selects PyTorch Device. Options are device = None or 'cpu' or 0 or '0' or '0,1,2,3'."""
    s = ''
    device = str(device).lower()
    for remove in 'cuda:', 'none', '(', ')', '[', ']', "'", ' ':
        device = device.replace(
            remove, ''
        )  # to string, 'cuda:0' -> '0' and '(0, 1)' -> '0,1'
    cpu = device == 'cpu'
    mps = device == 'mps'  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ[
            'CUDA_VISIBLE_DEVICES'
        ] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        if device == 'cuda':
            device = '0'
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        os.environ[
            'CUDA_VISIBLE_DEVICES'
        ] = device  # set environment variable - must be before assert is_available()
        if not (
            torch.cuda.is_available()
            and torch.cuda.device_count() >= len(device.replace(',', ''))
        ):
            LOGGER.info(s)
            install = (
                'See https://pytorch.org/get-started/locally/ for up-to-date torch install instructions if no '
                'CUDA devices are seen by torch.\n'
                if torch.cuda.device_count() == 0
                else ''
            )
            raise ValueError(
                f"Invalid CUDA 'device={device}' requested."
                f" Use 'device=cpu' or pass valid CUDA device(s) if available,"
                f" i.e. 'device=0' or 'device=0,1,2,3' for Multi-GPU.\n"
                f'\ntorch.cuda.is_available(): {torch.cuda.is_available()}'
                f'\ntorch.cuda.device_count(): {torch.cuda.device_count()}'
                f"\nos.environ['CUDA_VISIBLE_DEVICES']: {visible}\n"
                f'{install}'
            )

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = (
            device.split(',') if device else '0'
        )  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if (
            n > 1 and batch > 0 and batch % n != 0
        ):  # check batch_size is divisible by device_count
            raise ValueError(
                f"'batch={batch}' must be a multiple of GPU count {n}. Try 'batch={batch // n * n}' or "
                f"'batch={batch // n * n + n}', the nearest batch sizes evenly divisible by {n}."
            )
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
        arg = 'cuda:0'
    elif (
        mps
        and getattr(torch, 'has_mps', False)
        and torch.backends.mps.is_available()
        and TORCH_2_0
    ):
        # Prefer MPS if available
        s += 'MPS\n'
        arg = 'mps'
    else:  # revert to CPU
        s += 'CPU\n'
        arg = 'cpu'

    if verbose and RANK == -1:
        LOGGER.info(s if newline else s.rstrip())
    return torch.device(arg)


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """Decorator to make all processes in distributed training wait for each local_master to do something."""
    initialized = (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    if initialized and local_rank not in (-1, 0):
        dist.barrier(device_ids=[local_rank])
    yield
    if initialized and local_rank == 0:
        dist.barrier(device_ids=[0])


def one_cycle(y1=0.0, y2=1.0, steps=100):
    """Returns a lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf."""
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


class Profile(contextlib.ContextDecorator):
    """
    YOLOv8 Profile class.
    Usage: as a decorator with @Profile() or as a context manager with 'with Profile():'
    """

    def __init__(self, t=0.0):
        """
        Initialize the Profile class.

        Args:
            t (float): Initial time. Defaults to 0.0.
        """
        self.t = t
        self.cuda = torch.cuda.is_available()

    def __enter__(self):
        """
        Start timing.
        """
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        """
        Stop timing.
        """
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        """
        Get current time.
        """
        if self.cuda:
            torch.cuda.synchronize()
        return time.time()


def model_info(model, detailed=False, verbose=True, imgsz=640):
    """Model information. imgsz may be int or list, i.e. imgsz=640 or imgsz=[640, 320]."""
    if not verbose:
        return None
    n_p = get_num_params(model)  # number of parameters
    n_g = get_num_gradients(model)  # number of gradients
    n_l = len(list(model.modules()))  # number of layers
    if detailed:
        LOGGER.info(
            f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}"
        )
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            LOGGER.info(
                '%5g %40s %9s %12g %20s %10.3g %10.3g %10s',
                i,
                name,
                p.requires_grad,
                p.numel(),
                list(p.shape),
                p.mean(),
                p.std(),
                p.dtype,
            )

    fused = ' (fused)' if getattr(model, 'is_fused', lambda: False)() else ''
    yaml_file = getattr(model, 'yaml_file', '') or getattr(model, 'yaml', {}).get(
        'yaml_file', ''
    )
    model_name = Path(yaml_file).stem.replace('yolo', 'YOLO') or 'Model'
    LOGGER.info(
        f'{model_name} summary{fused}: {n_l} layers, {n_p} parameters, {n_g} gradients'
    )
    return n_l, n_p, n_g


def get_num_params(model):
    """Return the total number of parameters in a YOLO model."""
    return sum(x.numel() for x in model.parameters())


def get_num_gradients(model):
    """Return the total number of parameters with gradients in a YOLO model."""
    return sum(x.numel() for x in model.parameters() if x.requires_grad)


def fuse_blocks(model: torch.nn.Module) -> nn.Module:
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'fuse'):
            module.fuse()
    return model
