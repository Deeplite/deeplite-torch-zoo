#!/bin/bash
# https://github.com/rwightman/pytorch-image-models
NUM_PROC=$1
shift
torchrun --nproc_per_node=$NUM_PROC train.py "$@"
