#!/bin/sh

CUDA_VISIBLE_DEVICES=0,1,2,3 \
#python -m torch.distributed.launch --nproc_per_node 4 \
#  training_scripts/sparse_train.py --dataset cifar100 \
python sparse_train.py --dataset cifar100 \
                --imgsz 32 --data-root ./results \
                --model resnet18 --batch-size 128 \
                --epochs 10 --pretrained
