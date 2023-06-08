import hashlib
import os
import math
import logging.config

import torch

from ultralytics.yolo.utils import colorstr

import deeplite_torch_zoo


KB_IN_MB_COUNT = 1024
LOGGING_NAME = 'deeplite-torch-zoo'


def set_logging(name=LOGGING_NAME, verbose=True):
    # sets up logging for the given name
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig(
        {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {name: {'format': '%(message)s'}},
            'handlers': {
                name: {
                    'class': 'logging.StreamHandler',
                    'formatter': name,
                    'level': level,
                }
            },
            'loggers': {
                name: {
                    'level': level,
                    'handlers': [name],
                    'propagate': False,
                }
            },
        }
    )


set_logging(LOGGING_NAME)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally


def make_divisible(x, divisor):
    """
    Returns the nearest number that is divisible by the given divisor.

    Args:
        x (int): The number to make divisible.
        divisor (int) or (torch.Tensor): The divisor.

    Returns:
        (int): The nearest number divisible by the divisor.
    """
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


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
