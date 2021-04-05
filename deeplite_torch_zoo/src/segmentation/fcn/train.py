import argparse
import datetime
import math
import os
import os.path as osp
import shutil
from distutils.version import LooseVersion

import fcn
import imageio
import numpy as np
import pytz
import scipy.misc
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

from deeplite_torch_zoo.wrappers.wrapper import get_data_splits_by_name
from torchfcn.models import *
from deeplite_torch_zoo.src.segmentation.fcn.utils import label_accuracy_score
from deeplite_torch_zoo.src.segmentation.models.utils.optimizer import \
    create_optimizer


def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    if LooseVersion(torch.__version__) < LooseVersion("0.3"):
        # ==0.2.X
        log_p = F.log_softmax(input)
    else:
        # >=0.3
        log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss


class Trainer(object):
    def __init__(
        self,
        cuda,
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        out,
        epochs=100,
        size_average=False,
    ):
        self.cuda = cuda
        self.epochs = epochs

        self.model = model
        self.optim = optimizer
        self.scheduler = scheduler

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.timestamp_start = datetime.datetime.now(pytz.timezone("Asia/Tokyo"))
        self.size_average = size_average

        self.interval_validate = 5

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            "epoch",
            "iteration",
            "train/loss",
            "train/acc",
            "train/acc_cls",
            "train/mean_iu",
            "train/fwavacc",
            "valid/loss",
            "valid/acc",
            "valid/acc_cls",
            "valid/mean_iu",
            "valid/fwavacc",
            "elapsed_time",
        ]
        if not osp.exists(osp.join(self.out, "log.csv")):
            with open(osp.join(self.out, "log.csv"), "w") as f:
                f.write(",".join(self.log_headers) + "\n")

        self.epoch = 0
        self.iteration = 0
        self.best_mean_iu = 0

    def validate(self):
        training = self.model.training
        self.model.eval()

        n_class = self.val_loader.dataset.n_classes

        val_loss = 0
        visualizations = []
        label_trues, label_preds = [], []
        for batch_idx, (data, target, _) in tqdm(enumerate(self.val_loader)):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            score = self.model(data)

            loss = cross_entropy2d(score, target, size_average=self.size_average)
            loss_data = float(loss.item())
            if np.isnan(loss_data):
                raise ValueError("loss is nan while validating")
            val_loss += loss_data / len(data)

            imgs = data.data.cpu()
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu()
            for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                # img, lt = self.val_loader.dataset.untransform(img, lt)
                label_trues.append(lt.numpy())
                label_preds.append(lp)
                # if len(visualizations) < 9:
                #    viz = fcn.utils.visualize_segmentation(
                #        lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class)
                #    visualizations.append(viz)
        acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(
            label_trues, label_preds, n_class
        )
        metrics = [acc, acc_cls, mean_iu, fwavacc]

        # out = osp.join(self.out, 'visualization_viz')
        # if not osp.exists(out):
        #    os.makedirs(out)
        # out_file = osp.join(out, 'iter%012d.jpg' % self.iteration)
        # scipy.misc.imsave(out_file, fcn.utils.get_tile_image(visualizations))
        # imageio.imwrite(out_file, fcn.utils.get_tile_image(visualizations))

        val_loss /= len(self.val_loader)
        print(
            "acc {:0.3f}, acc_cls {:0.3f}, mean_iu {:0.3f}, fwavacc {:0.3f}, val loss {}".format(
                metrics[0], metrics[1], metrics[2], metrics[3], val_loss
            )
        )

        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu
        torch.save(
            {
                "epoch": self.epoch,
                "iteration": self.iteration,
                "arch": self.model.__class__.__name__,
                "optim_state_dict": self.optim.state_dict(),
                "model_state_dict": self.model.state_dict(),
                "best_mean_iu": self.best_mean_iu,
            },
            osp.join(self.out, "checkpoint.pth.tar"),
        )
        if is_best:
            shutil.copy(
                osp.join(self.out, "checkpoint.pth.tar"),
                osp.join(self.out, "model_best.pth.tar"),
            )

        if training:
            self.model.train()

    def train_epoch(self):
        self.model.train()

        n_class = self.train_loader.dataset.n_classes
        metrics = []
        train_loss = 0

        for batch_idx, (data, target, _) in tqdm(enumerate(self.train_loader)):
            assert self.model.training

            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optim.zero_grad()
            score = self.model(data)

            loss = cross_entropy2d(score, target, size_average=self.size_average)
            loss /= len(data)
            loss_data = float(loss.item())
            train_loss += loss_data
            if np.isnan(loss_data):
                raise ValueError("loss is nan while training")
            loss.backward()
            self.optim.step()

            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu().numpy()
            acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(
                lbl_true, lbl_pred, n_class=n_class
            )
            metrics.append([acc, acc_cls, mean_iu, fwavacc])
        train_loss /= len(self.train_loader)
        metrics = np.mean(metrics, axis=0)
        # print(f'train loss {train_loss}')
        # if isinstance(metrics, list) and len(metrics == 4):
        print(
            "acc {:0.3f}, acc_cls {:0.3f}, mean_iu {:0.3f}, fwavacc {:0.3f}, train loss {}".format(
                metrics[0], metrics[1], metrics[2], metrics[3], train_loss
            )
        )

        if self.epoch % self.interval_validate == 0:
            self.validate()

    def train(self):
        for epoch in range(0, self.epochs):
            self.epoch = epoch
            print("epoch {}/{}".format(epoch, self.epochs))
            self.train_epoch()
            self.scheduler.step()


def get_args():
    parser = argparse.ArgumentParser(
        description="Train the FCN on images and target masks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-e",
        "--epochs",
        metavar="E",
        type=int,
        default=100,
        help="Number of epochs",
        dest="epochs",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        metavar="B",
        type=int,
        nargs="?",
        default=2,
        help="Batch size",
        dest="batchsize",
    )
    parser.add_argument(
        "-l",
        "--learning-rate",
        metavar="LR",
        type=float,
        nargs="?",
        default=0.0001,
        help="Learning rate",
        dest="lr",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}".format(device))

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N

    data_splits = get_data_splits_by_name(
        data_root="data/VOC/VOCdevkit/VOC2012/",
        dataset_name="voc",
        model_name="fcn",
        batch_size=args.batchsize,
        num_workers=1,
    )
    net = FCN32s(n_class=data_splits["train"].dataset.n_classes)

    net.to(device=device)
    optimizer, scheduler = create_optimizer(
        net.parameters(), mode="adam", base_lr=args.lr
    )
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        Trainer(
            model=net,
            cuda=True,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=data_splits["train"],
            val_loader=data_splits["test"],
            out="data/weight/segmentation/fcn32s/pascal/",
            epochs=args.epochs,
        ).train()
    except KeyboardInterrupt:
        torch.save(net.state_dict(), "INTERRUPTED.pth")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
