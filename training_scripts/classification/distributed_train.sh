#!/bin/bash
# https://github.com/rwightman/pytorch-image-models
NUM_PROC=$1
shift
python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC train.py "$@"
