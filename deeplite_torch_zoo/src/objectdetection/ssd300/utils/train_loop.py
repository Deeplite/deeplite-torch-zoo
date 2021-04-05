# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

import torch

# from deeplite_torch_zoo import _C as C


def train_loop(
    model,
    loss_func,
    epoch,
    optim,
    train_dataloader,
    val_dataloader,
    iteration,
    logger,
    args,
    mean,
    std,
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    avg_loss = 0
    #    for nbatch, (images, targets, bbox_offsets, _) in enumerate(train_dataloader):
    for nbatch, (images, targets, labels_length, _) in enumerate(train_dataloader):

        images = images.to(device)

        # images, bboxes = C.random_horiz_flip(images, bboxes, bbox_offsets, 0.5, False)
        # if bbox_offsets[-1].item() == 0:
        #    print("No labels in batch")
        #    continue

        images.sub_(mean).div_(std)

        ploc, plabel = model(images)

        loss = loss_func.compute_loss(
            images.shape, targets, labels_length, ploc, plabel, device=device
        )

        avg_loss = (nbatch * avg_loss + loss.item()) / (nbatch + 1)
        logger.update_iter(epoch, iteration, avg_loss)

        # loss scaling
        loss.backward()

        if args.warmup is not None:
            warmup(optim, args.warmup, iteration, args.learning_rate)

        optim.step()
        optim.zero_grad()
        iteration += 1
    return iteration


def warmup(optim, warmup_iters, iteration, base_lr):
    if iteration < warmup_iters:
        new_lr = 1.0 * base_lr / warmup_iters * iteration
        for param_group in optim.param_groups:
            param_group["lr"] = new_lr


def load_checkpoint(model, checkpoint):
    """
    Load model from checkpoint.
    """
    print("loading model checkpoint", checkpoint)
    od = torch.load(checkpoint)

    # remove proceeding 'N.' from checkpoint that comes from DDP wrapper
    saved_model = od["model"]
    model.load_state_dict(saved_model)


def tencent_trick(model):
    """
    Divide parameters into 2 groups.
    First group is BNs and all biases.
    Second group is the remaining model's parameters.
    Weight decay will be disabled in first group (aka tencent trick).
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{"params": no_decay, "weight_decay": 0.0}, {"params": decay}]
