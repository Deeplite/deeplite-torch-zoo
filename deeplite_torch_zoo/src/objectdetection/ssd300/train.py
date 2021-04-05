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

import os
import time
from argparse import ArgumentParser

import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR

import deeplite_torch_zoo.src.objectdetection.configs.ssd_hyp_config as hyp_cfg
import deeplite_torch_zoo.src.objectdetection.configs.voc_config as cfg
from deeplite_torch_zoo.wrappers.wrapper import get_data_splits_by_name
from deeplite_torch_zoo.wrappers.models import yolo_eval_func
from deeplite_torch_zoo.src.objectdetection.ssd300.model.ssd300 import SSD300
from deeplite_torch_zoo.src.objectdetection.ssd300.model.ssd300_loss import Loss
from deeplite_torch_zoo.src.objectdetection.ssd300.utils.logger import (BenchLogger, Logger)
from deeplite_torch_zoo.src.objectdetection.ssd300.utils.train_loop import (load_checkpoint, tencent_trick, train_loop)
from deeplite_torch_zoo.src.objectdetection.ssd300.utils.utils import (Encoder, dboxes300_coco)


def generate_mean_std(args):
    mean_val = [0.485, 0.456, 0.406]
    std_val = [0.229, 0.224, 0.225]

    mean = torch.tensor(mean_val).to(args.device)
    std = torch.tensor(std_val).to(args.device)

    view = [1, len(mean_val), 1, 1]

    mean = mean.view(*view)
    std = std.view(*view)

    if args.amp:
        mean = mean.half()
        std = std.half()

    return mean, std


def make_parser():
    parser = ArgumentParser(description="Train Single Shot MultiBox Detector")
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default="deeplite_torch_zoo/data/VOC/VOCdevkit",
        required=False,
        help="path to test and training data files",
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=100, help="number of epochs for training"
    )
    parser.add_argument(
        "--batch-size",
        "--bs",
        type=int,
        default=8,
        help="number of examples for each iteration",
    )
    parser.add_argument(
        "--eval-batch-size",
        "--ebs",
        type=int,
        default=4,
        help="number of examples for each evaluation iteration",
    )
    parser.add_argument(
        "--no-cuda", default=False, action="store_true", help="use available GPUs"
    )
    parser.add_argument(
        "--seed", "-s", type=int, help="manually set random seed for torch"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="path to model checkpoint file"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="deeplite_torch_zoo/weight/",
        help="save model checkpoints here",
    )
    parser.add_argument(
        "--mode", type=str, default="training", choices=["training", "evaluation"]
    )
    parser.add_argument(
        "--evaluation", type=int, default=10, help="epochs at which to evaluate"
    )
    parser.add_argument(
        "--multistep",
        nargs="*",
        type=int,
        default=[80, 90],
        help="epochs at which to decay learning rate",
    )

    # Hyperparameters
    parser.add_argument(
        "--learning-rate", "--lr", type=float, default=2.6e-3, help="learning rate"
    )
    parser.add_argument(
        "--momentum",
        "-m",
        type=float,
        default=0.9,
        help="momentum argument for SGD optimizer",
    )
    parser.add_argument(
        "--weight-decay",
        "--wd",
        type=float,
        default=0.0005,
        help="momentum argument for SGD optimizer",
    )

    parser.add_argument("--profile", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)

    parser.add_argument(
        "--backbone",
        type=str,
        default="vgg16",
        choices=[
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
            "mobilenet_v2",
            "vgg16",
            "vgg16_bn",
        ],
    )
    parser.add_argument(
        "--pretrained",
        type=bool,
        default=True,
        help="if true loads the pretrained weights of the backbone layers",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--amp", action="store_true")

    parser.add_argument(
        "--crop-val",
        type=bool,
        default=False,
        help="if true loads the pretrained weights of the backbone layers",
    )

    return parser


def train(train_loop_func, logger, args):
    # Check that GPUs are actually available

    if args.seed is None:
        args.seed = np.random.randint(1e4)

    print("Using seed = {}".format(args.seed))
    torch.manual_seed(args.seed)
    np.random.seed(seed=args.seed)

    dataset_splits = get_data_splits_by_name(
        data_root=args.data,
        dataset_name='voc',
        model_name='yolo',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_classes=cf.DATA["NUM"],
        img_size=hyp_cfg.TRAIN["TRAIN_IMG_SIZE"]
    )
    train_loader = dataset_splits["train"]
    val_dataloader = dataset_splits["val"]
    dboxes = dboxes300_coco()
    ssd300 = SSD300(backbone=args.backbone, pretrained=args.pretrained)
    start_epoch = 0
    iteration = 0
    loss_func = Loss(dboxes)

    ssd300.to(args.device)
    loss_func.to(args.device)

    optimizer = torch.optim.SGD(
        tencent_trick(ssd300),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = MultiStepLR(optimizer=optimizer, milestones=args.multistep, gamma=0.1)
    if args.amp:
        ssd300, optimizer = amp.initialize(ssd300, optimizer, opt_level="O2")

    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            load_checkpoint(ssd300, args.checkpoint)
            checkpoint = torch.load(
                args.checkpoint,
                map_location=lambda storage, loc: storage.cuda(
                    torch.cuda.current_device()
                ),
            )
            start_epoch = checkpoint["epoch"]
            iteration = checkpoint["iteration"]
            scheduler.load_state_dict(checkpoint["scheduler"])
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            print("Provided checkpoint is not path to a file")
            return

    total_time = 0

    if args.mode == "evaluation":
        Aps = yolo_eval_func(
            ssd300,
            os.path.join(args.data, "VOC2007"),
            _set="voc",
            img_size=hyp_cfg.TEST["TEST_IMG_SIZE"],
            device=args.device,
            net="ssd300",
        )
        mAP = Aps["mAP"]
        print("Model precision {} mAP".format(mAP))
        return

    mean, std = generate_mean_std(args)

    for epoch in range(start_epoch, args.epochs):
        start_epoch_time = time.time()
        scheduler.step()
        iteration = train_loop_func(
            ssd300,
            loss_func,
            epoch,
            optimizer,
            train_loader,
            val_dataloader,
            iteration,
            logger,
            args,
            mean,
            std,
        )
        end_epoch_time = time.time() - start_epoch_time
        total_time += end_epoch_time

        logger.update_epoch_time(epoch, end_epoch_time)

        if epoch % args.evaluation == 0:
            Aps = yolo_eval_func(
                ssd300,
                os.path.join(args.data, "VOC2007"),
                _set="voc",
                img_size=hyp_cfg.TEST["TEST_IMG_SIZE"],
                device=args.device,
                net="ssd300",
            )
            mAP = Aps["mAP"]
            logger.update_epoch(epoch, mAP)

        if args.save_path is not None and epoch % args.evaluation == 0:
            print("saving model...")
            obj = {
                "epoch": epoch + 1,
                "iteration": iteration,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }

            obj["model"] = ssd300.state_dict()
            torch.save(obj, "{}/epoch_{}.pt".format(args.save_path, epoch))
    print("total training time: {}".format(total_time))


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    if args.save_path is not None:
        args.save_path = "{}/ssd300/voc/{}".format(args.save_path, args.backbone)
        os.makedirs(args.save_path, exist_ok=True)

    train_loop_func = train_loop
    logger = Logger("Training logger", print_freq=100)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    args.device = device
    train(train_loop_func, logger, args)
