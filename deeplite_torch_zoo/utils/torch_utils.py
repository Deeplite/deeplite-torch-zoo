import os
import random
from contextlib import contextmanager

import numpy as np

import torch
import torchvision
import torch.nn as nn

from torch.hub import load_state_dict_from_url

from ultralytics.yolo.utils.checks import check_version
from ultralytics.yolo.utils.torch_utils import fuse_conv_and_bn, model_info, scale_img

import deeplite_torch_zoo
from deeplite_torch_zoo.utils import LOGGER, get_file_hash


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


def load_pretrained_weights(model, checkpoint_url, device):
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
