import argparse
import os
import re
import random
import math
import functools
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp

import numpy as np
from pycocotools.coco import COCO


from deeplite_torch_zoo.wrappers.wrapper import get_data_splits_by_name
import deeplite_torch_zoo.src.objectdetection.configs.hyps.hyp_config_default as hyp_cfg_scratch
import deeplite_torch_zoo.src.objectdetection.configs.hyps.hyp_config_finetune as hyp_cfg_finetune
import deeplite_torch_zoo.src.objectdetection.configs.hyps.hyp_config_lisa as hyp_cfg_lisa

from deeplite_torch_zoo.src.objectdetection.yolov5.utils.torch_utils import select_device, init_seeds
from deeplite_torch_zoo.wrappers.models import yolo3, yolo4, yolo5, yolo5_6
from deeplite_torch_zoo.wrappers.eval import get_eval_func
from deeplite_torch_zoo.src.objectdetection.yolov5.models.yolov5_loss import \
    YoloV5Loss
from deeplite_torch_zoo.src.objectdetection.datasets.coco import SubsampledCOCO

from deeplite_torch_zoo.wrappers.models import YOLOV3_MODELS, YOLOV4_MODELS, YOLOV5_MODELS


YOLO_MODEL_NAMES = YOLOV3_MODELS + YOLOV4_MODELS + YOLOV5_MODELS

DATASET_TO_HP_CONFIG_MAP = {
    'lisa': hyp_cfg_lisa,
}

for dataset_name in ('voc', 'coco', 'wider_face', 'person_detection',
    'car_detection', 'voc07'):
    DATASET_TO_HP_CONFIG_MAP[dataset_name] = hyp_cfg_scratch

HP_CONFIG_MAP = {
    'scratch': hyp_cfg_scratch,
    'finetune': hyp_cfg_finetune,
}

class Trainer(object):
    def __init__(self, weight_path, resume, gpu_id):
        init_seeds(0)

        self.model_name = opt.net
        assert self.model_name in YOLO_MODEL_NAMES

        if opt.hp_config is None:
            for dataset_name in DATASET_TO_HP_CONFIG_MAP:
                if dataset_name in opt.dataset_type:
                    self.hyp_config = DATASET_TO_HP_CONFIG_MAP[dataset_name]
        else:
            self.hyp_config = HP_CONFIG_MAP[opt.hp_config]

        self.device = select_device(gpu_id)
        self.start_epoch = 0
        self.best_mAP = 0.0
        self.epochs = self.hyp_config.TRAIN["EPOCHS"]

        self.multi_scale_train = self.hyp_config.TRAIN["MULTI_SCALE_TRAIN"]

        dataset_splits = get_data_splits_by_name(
            data_root=opt.img_dir,
            dataset_name=opt.dataset_type,
            model_name=self.model_name,
            batch_size=opt.batch_size,
            num_workers=opt.n_cpu,
            img_size=self.hyp_config.TRAIN["TRAIN_IMG_SIZE"],
        )

        self.train_dataloader = dataset_splits["train"]
        self.train_dataset = self.train_dataloader.dataset
        self.val_dataloader = dataset_splits["val"]
        self.num_classes = self.train_dataset.num_classes
        self.weight_path = weight_path / self.model_name / "{}_{}_cls".format(opt.dataset_type, self.num_classes)
        Path(self.weight_path).mkdir(parents=True, exist_ok=True)

        self.pretraining_source_dataset = opt.pretraining_source_dataset
        self.model = self._get_model()

        self.criterion = self._get_loss()
        if resume:
            self.__load_model_weights(weight_path, resume)

        self.optimizer, self.scheduler, self.warmup_training_callback = make_od_optimizer(self.model,
            self.epochs, hyp_config=self.hyp_config)

        self.scaler = amp.GradScaler()

    def _get_model(self):
        net_name_to_model_fn_map = {
            "^yolov3$": yolo3,
            "^yolov5[smlx]$": yolo5,
            "^yolov5_6[nsmlx]$": yolo5_6,
            "^yolov5_6[nsmlx]a$": yolo5_6,
            "^yolov5_6[nsmlx]_relu$": functools.partial(yolo5_6, activation_type='relu'),
            "^yolov4[smlx]$": yolo4,
        }
        default_model_fn_args = {
            "pretrained": opt.pretrained,
            "num_classes": self.num_classes,
            "device": self.device,
            "progress": True,
            "dataset_name": self.pretraining_source_dataset,
        }
        for net_name, model_fn in net_name_to_model_fn_map.items():
            if re.match(net_name, self.model_name):
                return model_fn(self.model_name, **default_model_fn_args)

    def _get_loss(self):
        loss_kwargs = {
            'model': self.model,
            'num_classes': self.num_classes,
            'device': self.device,
            'hyp_cfg': self.hyp_config,
        }
        return YoloV5Loss(**loss_kwargs)

    def __load_model_weights(self, weight_path, resume):
        if resume:
            last_weight = self.weight_path / "last.pt"
            chkpt = torch.load(last_weight, map_location=self.device)
            self.model.load_state_dict(chkpt["model"])

            self.start_epoch = chkpt["epoch"] + 1
            if chkpt["optimizer"] is not None:
                self.optimizer.load_state_dict(chkpt["optimizer"])
                self.best_mAP = chkpt["best_mAP"]
            del chkpt
        else:
            self.model.load_darknet_weights(weight_path)

    def __save_model_weights(self, epoch, mAP):
        if mAP > self.best_mAP:
            self.best_mAP = mAP
        best_weight = self.weight_path / "best.pt"
        last_weight = self.weight_path / "last.pt"
        chkpt = {
            "epoch": epoch,
            "best_mAP": self.best_mAP,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(chkpt, last_weight)

        if self.best_mAP == mAP:
            torch.save(chkpt["model"], best_weight)

        if epoch > 0 and epoch % opt.checkpoint_save_freq == 0:
            torch.save(
                chkpt,
                os.path.join(
                    os.path.split(self.weight_path)[0], "backup_epoch%g.pt" % epoch
                ),
            )
        del chkpt

    def evaluate(self):
        self.model.eval()
        eval_func = get_eval_func(opt.dataset_type)
        test_set = opt.img_dir
        gt = None
        if opt.dataset_type in ("voc", "voc07"):
            test_set = opt.img_dir / "VOC2007"
        elif opt.dataset_type == "coco":
            gt = COCO(opt.img_dir / "annotations/instances_val2017.json")
        elif opt.dataset_type == "car_detection":
            gt = SubsampledCOCO(opt.img_dir / "annotations/instances_val2017.json",
                subsample_categories=['car'])

        Aps = eval_func(self.model, test_set, gt=gt, num_classes=self.num_classes,
                        _set=opt.dataset_type, device=self.device, net=opt.net,
                        img_size=opt.test_img_res, progressbar=True)
        return Aps

    def train(self):
        print(self.model)
        print("The number of samples in the train dataset split: {}".format(len(self.train_dataset)))

        num_batches = len(self.train_dataloader)

        if opt.eval_before_train:
            Aps = self.evaluate()
            print(f"Initial mAP values: {Aps}")

        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()

            mloss = torch.zeros(4)
            self.optimizer.zero_grad()
            for i, (imgs, targets, labels_length, _) in enumerate(self.train_dataloader):
                num_iter = i + epoch * num_batches
                self.warmup_training_callback(self.train_dataloader, epoch, num_iter)

                with amp.autocast():
                    imgs = imgs.to(self.device)
                    p, p_d = self.model(imgs)
                    loss, loss_giou, loss_conf, loss_cls = self.criterion(
                        p, p_d, targets, labels_length, imgs.shape[-1]
                    )
                    # Update running mean of tracked metrics
                    loss_items = torch.tensor([loss_giou, loss_conf, loss_cls, loss])
                    mloss = (mloss * i + loss_items) / (i + 1)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)  # optimizer.step
                self.scaler.update()
                self.optimizer.zero_grad()

                print(f"\repoch {epoch}/{self.epochs} - Iteration: {i}/{len(self.train_dataloader)}, " \
                    f" loss: giou {mloss[0]:0.4f}    conf {mloss[1]:0.4f}    cls {mloss[2]:0.4f}    " \
                    f"loss {mloss[3]:0.4f}", end="")

                # multi-scale training (320-608 pixel resolution)
                if self.multi_scale_train:
                    self.train_dataset._img_size = random.choice(range(10, 20)) * 32

            self.scheduler.step()

            mAP = 0
            if epoch % opt.eval_freq == 0:
                Aps = self.evaluate()
                mAP = Aps["mAP"]
                self.__save_model_weights(epoch, mAP)
                print("best mAP : %g" % (self.best_mAP))


def make_od_optimizer(model, epochs, hyp_config=None, hyp_config_name=None, linear_lr=False):
    if hyp_config is None:
        hyp_config = HP_CONFIG_MAP.get(hyp_config_name, hyp_cfg_scratch)

    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)
    optimizer = optim.SGD(g0, lr=hyp_config.TRAIN['lr0'], momentum=hyp_config.TRAIN['momentum'], nesterov=True)
    optimizer.add_param_group({'params': g1, 'weight_decay': hyp_config.TRAIN['weight_decay']})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g2})  # add g2 (biases)

    # Scheduler
    if linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp_config.TRAIN['lrf']) + hyp_config.TRAIN['lrf']  # linear
    else:
        lf = one_cycle(1, hyp_config.TRAIN['lrf'], epochs)  # cosine 1->hyp['lrf']

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    warmup_training_fn = functools.partial(warmup_training, optimizer=optimizer, scheduler=scheduler,
        hyp_config=hyp_config)
    return optimizer, scheduler, warmup_training_fn


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def warmup_training(train_dataloder, epoch, iter_number, optimizer=None, scheduler=None, hyp_config=None):
    num_batches_per_epoch = len(train_dataloder)
    warmup_iters_number = hyp_config.TRAIN['warmup_epochs'] * num_batches_per_epoch
    if iter_number < warmup_iters_number:
        xi = [0, warmup_iters_number]  # x interp
        for j, x in enumerate(optimizer.param_groups):
            # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
            x['lr'] = np.interp(iter_number, xi, [hyp_config.TRAIN['warmup_bias_lr'] if j == 2 else 0.0,
                x['initial_lr'] * scheduler.lr_lambdas[0](epoch)])
            if 'momentum' in x:
                x['momentum'] = np.interp(iter_number, xi, [hyp_config.TRAIN['warmup_momentum'],
                    hyp_config.TRAIN['momentum']])

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", dest="img_dir", type=Path, default="/neutrino/datasets/VOCdevkit", help="The path to the folder containing images to be detected or trained.")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=10, help="The number of samples in one batch during training or inference.")
    parser.add_argument("--eval-freq", dest="eval_freq", type=int, default=10, help="Evaluation run frequency (in training epochs).")
    parser.add_argument("--weight_path", type=Path, default="models/", help="where weights should be stored")
    parser.add_argument("--checkpoint_save_freq", type=int, default=10, help="Checkpoint dump frequency in training epochs")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training flag")
    parser.add_argument("--pretrained", action="store_true", default=False, help="Train the model from scratch if False")
    parser.add_argument("--pretraining_source_dataset", type=str, default="voc_20", help="Load pretrained weights fine-tuned on the specified dataset ('voc_20' or 'coco_80')")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument("--n-cpu", dest="n_cpu", type=int, default=4, help="The number of cpu threads to use during batch generation.")
    parser.add_argument("--dataset", dest="dataset_type", type=str, default="voc",
        choices=["coco", "voc", "lisa", "lisa_full", "lisa_subset11", "wider_face", "person_detection", "voc07", "car_detection", "person_pet_vehicle_detection"],
        help="Name of the dataset to train/validate on",
    )
    parser.add_argument("--net", dest="net", type=str, default="yolov4m", help="The type of the network used. Currently support 'yolo3', 'yolo4' and 'yolo5'")
    parser.add_argument("--hp_config", dest="hp_config", type=str, default=None, help="The hyperparameter configuration name to use. Available options: 'scratch', 'finetune'")
    parser.add_argument("--test_img_res", dest="test_img_res", type=int, default=448, help="Image resolution to use during model testing")
    parser.add_argument("--eval_before_train", dest="eval_before_train", action="store_true", default=False, help="Run model evaluation before training")
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    Trainer(weight_path=opt.weight_path, resume=opt.resume, gpu_id=opt.gpu_id).train()