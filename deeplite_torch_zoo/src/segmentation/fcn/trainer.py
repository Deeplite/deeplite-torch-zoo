import datetime
import math
import os
import os.path as osp
import shutil

import fcn
import imageio
import numpy as np
import pytz
import torch
import tqdm
import utils
from solver import solver


class Trainer(solver):
    def __init__(self, data_loader, opts):
        super(Trainer, self).__init__(data_loader, opts)
        self.cuda = opts.cuda
        self.opts = opts
        self.train_loader = data_loader[0]
        self.val_loader = data_loader[1]

        if opts.mode in ["val", "demo"]:
            return

        self.timestamp_start = datetime.datetime.now(pytz.timezone("America/Bogota"))

        self.interval_validate = opts.cfg.get(
            "interval_validate", len(self.train_loader)
        )
        if self.interval_validate is None:
            self.interval_validate = len(self.train_loader)

        self.out = opts.out
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
        self.max_iter = opts.cfg["max_iteration"]
        self.best_mean_iu = 0

    def validate(self):
        # import matplotlib.pyplot as plt
        training = self.model.training
        self.model.eval()

        n_class = len(self.val_loader.dataset.class_names)

        val_loss = 0
        visualizations = []
        label_trues, label_preds = [], []
        with torch.no_grad():
            for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.val_loader),
                total=len(self.val_loader),
                desc="Valid iteration=%d" % self.iteration,
                ncols=80,
                leave=False,
            ):
                data, target = data.to(self.cuda), target.to(self.cuda)
                score = self.model(data)

                loss = self.cross_entropy2d(score, target)
                if np.isnan(float(loss.item())):
                    raise ValueError("loss is nan while validating")
                val_loss += float(loss.item()) / len(data)

                imgs = data.data.cpu()
                lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
                lbl_true = target.data.cpu()
                for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                    img, lt = self.val_loader.dataset.untransform(img, lt)
                    label_trues.append(lt)
                    label_preds.append(lp)
                    if len(visualizations) < 9:
                        viz = fcn.utils.visualize_segmentation(
                            lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class
                        )
                        visualizations.append(viz)
        metrics = utils.label_accuracy_score(label_trues, label_preds, n_class)

        out = osp.join(self.out, "visualization_viz")
        if not osp.exists(out):
            os.makedirs(out)
        out_file = osp.join(out, "iter%012d.jpg" % self.iteration)
        img_ = fcn.utils.get_tile_image(visualizations)
        imageio.imwrite(out_file, img_)
        # plt.imshow(imageio.imread(out_file))
        # plt.show()

        val_loss /= len(self.val_loader)
        print(
            "acc {:0.3f}, acc_cls {:0.3f}, mean_iu {:0.3f}, fwavacc {:0.3f}, val loss {}".format(
                metrics[0], metrics[1], metrics[2], metrics[3], val_loss
            )
        )

        with open(osp.join(self.out, "log.csv"), "a") as f:
            elapsed_time = (
                datetime.datetime.now(pytz.timezone("America/Bogota"))
                - self.timestamp_start
            ).total_seconds()
            log = (
                [self.epoch, self.iteration]
                + [""] * 5
                + [val_loss]
                + list(metrics)
                + [elapsed_time]
            )
            log = map(str, log)
            f.write(",".join(log) + "\n")

        mean_iu = metrics[2]
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

        n_class = len(self.train_loader.dataset.class_names)

        for batch_idx, (data, target) in tqdm.tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc="Train epoch=%d" % self.epoch,
            ncols=80,
            leave=False,
        ):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration

            if self.iteration % self.interval_validate == 0:
                self.validate()

            assert self.model.training

            data, target = data.to(self.cuda), target.to(self.cuda)
            self.optim.zero_grad()
            score = self.model(data)

            loss = self.cross_entropy2d(score, target)
            loss /= len(data)
            if np.isnan(float(loss.item())):
                raise ValueError("loss is nan while training")
            loss.backward()
            self.optim.step()

            metrics = []
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu().numpy()
            acc, acc_cls, mean_iu, fwavacc = utils.label_accuracy_score(
                lbl_true, lbl_pred, n_class=n_class
            )
            metrics.append((acc, acc_cls, mean_iu, fwavacc))
            metrics = np.mean(metrics, axis=0)

            with open(osp.join(self.out, "log.csv"), "a") as f:
                elapsed_time = (
                    datetime.datetime.now(pytz.timezone("America/Bogota"))
                    - self.timestamp_start
                ).total_seconds()
                log = (
                    [self.epoch, self.iteration]
                    + [loss.item()]
                    + metrics.tolist()
                    + [""] * 5
                    + [elapsed_time]
                )
                log = map(str, log)
                f.write(",".join(log) + "\n")

            if self.iteration >= self.max_iter:
                break

    def Train(self):
        max_epoch = int(math.ceil(1.0 * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch, desc="Train", ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break

    def Test(self):
        from utils import run_fromfile

        for image, label in self.val_loader:
            run_fromfile(
                self.model,
                image,
                self.opts.cuda,
                self.val_loader.dataset.untransform,
                val=True,
            )

    def Demo(self):
        import glob

        from utils import run_fromfile

        img_files = sorted(glob.glob("imgs/*.jpg"))
        for img in img_files:
            run_fromfile(
                self.model, img, self.opts.cuda, self.val_loader.dataset.transforms
            )
