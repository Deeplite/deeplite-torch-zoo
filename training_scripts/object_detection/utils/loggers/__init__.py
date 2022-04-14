# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Logging utils
"""

import os
from torch.utils.tensorboard import SummaryWriter
from utils.general import colorstr

LOGGERS = ('csv', 'tb')  # text-file, TensorBoard
RANK = int(os.getenv('RANK', -1))


class Loggers():
    # YOLOv5 Loggers class
    def __init__(self, save_dir=None, weights=None, opt=None, hyp=None, logger=None, include=LOGGERS):
        self.save_dir = save_dir
        self.weights = weights
        self.opt = opt
        self.hyp = hyp
        self.logger = logger  # for printing results to console
        self.include = include
        self.keys = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                     'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',  # metrics
                     'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                     'x/lr0', 'x/lr1', 'x/lr2']  # params
        for k in LOGGERS:
            setattr(self, k, None)  # init empty logger dictionary
        self.csv = True  # always log to csv

        # TensorBoard
        s = self.save_dir
        if 'tb' in self.include:
            prefix = colorstr('TensorBoard: ')
            self.logger.info(f"{prefix}Start with 'tensorboard --logdir {s.parent}', view at http://localhost:6006/")
            self.tb = SummaryWriter(str(s))

    def on_pretrain_routine_end(self):
        # Callback runs on pre-train routine end
        pass

    def on_train_batch_end(self, ni, model, imgs, targets, paths, plots, sync_bn):
        # Callback runs on train batch end
        pass

    def on_train_epoch_end(self, epoch):
        # Callback runs on train epoch end
        pass

    def on_val_image_end(self, pred, predn, path, names, im):
        # Callback runs on val image end
        pass

    def on_val_end(self):
        # Callback runs on val end
        pass

    def on_fit_epoch_end(self, vals, epoch, best_fitness, fi):
        # Callback runs at the end of each fit (train+val) epoch
        x = {k: v for k, v in zip(self.keys, vals)}  # dict
        if self.tb:
            for k, v in x.items():
                self.tb.add_scalar(k, v, epoch)

    def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
        # Callback runs on model save event
        pass

    def on_train_end(self, last, best, plots, epoch, results):
        # Callback runs on training end
        pass