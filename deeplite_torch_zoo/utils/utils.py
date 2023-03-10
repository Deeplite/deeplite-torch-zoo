import hashlib
import os
from contextlib import contextmanager

import torch
from torch.hub import load_state_dict_from_url

import deeplite_torch_zoo

KB_IN_MB_COUNT = 1024


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


def get_file_hash(filename, max_has_symbols=16, min_large_file_size_mb=1000):
    filesize_mb = os.path.getsize(filename) / (KB_IN_MB_COUNT * KB_IN_MB_COUNT)
    is_large_file = filesize_mb > min_large_file_size_mb

    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        if is_large_file:
            # Read and update hash string value in blocks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
            readable_hash = sha256_hash.hexdigest()
        else:
            bytes = f.read()  # read entire file as bytes
            readable_hash = hashlib.sha256(bytes).hexdigest()

    return readable_hash[:max_has_symbols]


def load_pretrained_weights(model, checkpoint_url, progress, device):
    pretrained_dict = load_state_dict_from_url(
        checkpoint_url,
        progress=progress,
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
    print(f'Loaded {len(pretrained_dict)}/{len(model_dict)} modules')


@contextmanager
def switch_train_mode(model, is_training=False):
    is_original_mode_training = model.training
    model.train(is_training)
    try:
        yield
    finally:
        model.train(is_original_mode_training)
