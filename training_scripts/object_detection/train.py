import argparse
import os
import hashlib
import random
import math
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter

from pycocotools.coco import COCO

from deeplite_torch_zoo import get_data_splits_by_name, create_model

import deeplite_torch_zoo.src.objectdetection.yolov5.configs.hyps.hyp_config_default as hyp_cfg_scratch
import deeplite_torch_zoo.src.objectdetection.yolov5.configs.hyps.hyp_config_finetune as hyp_cfg_finetune
import deeplite_torch_zoo.src.objectdetection.yolov5.configs.hyps.hyp_config_voc as hyp_cfg_voc
import deeplite_torch_zoo.src.objectdetection.yolov5.configs.hyps.hyp_config_lisa as hyp_cfg_lisa

from deeplite_torch_zoo.src.objectdetection.yolov5.utils.torch_utils import (
    select_device,
    init_seeds,
)
from deeplite_torch_zoo.wrappers.eval import get_eval_func
from deeplite_torch_zoo.src.objectdetection.yolov5.models.yolov5_loss import YoloV5Loss
from deeplite_torch_zoo.src.objectdetection.yolov5.utils.scheduler import CosineDecayLR
from deeplite_torch_zoo.src.objectdetection.datasets.coco import SubsampledCOCO


DATASET_TO_HP_CONFIG_MAP = {
    "lisa": hyp_cfg_lisa,
    "voc": hyp_cfg_voc,
    "voc07": hyp_cfg_voc,
}

for dataset_name in (
    "coco",
    "wider_face",
    "person_detection",
    "car_detection",
):
    DATASET_TO_HP_CONFIG_MAP[dataset_name] = hyp_cfg_scratch

HP_CONFIG_MAP = {
    "scratch": hyp_cfg_scratch,
    "finetune": hyp_cfg_finetune,
}


class Trainer(object):
    def __init__(self, weight_path, resume, gpu_id):
        init_seeds(0)

        self.model_name = opt.net

        if opt.hp_config is None:
            for dataset_name in DATASET_TO_HP_CONFIG_MAP:
                if dataset_name in opt.dataset_type:
                    self.hyp_config = DATASET_TO_HP_CONFIG_MAP[dataset_name]
        else:
            self.hyp_config = HP_CONFIG_MAP[opt.hp_config]

        self.device = select_device(gpu_id)
        self.start_epoch = 0
        self.best_mAP = 0.0
        self.epochs = self.hyp_config.TRAIN["epochs"] if not opt.epochs else opt.epochs

        self.multi_scale_train = self.hyp_config.TRAIN["multi_scale_train"]

        train_img_size = self.hyp_config.TRAIN["train_img_size"] if not opt.train_img_res \
            else opt.train_img_res

        dataset_splits = get_data_splits_by_name(
            data_root=opt.img_dir,
            dataset_name=opt.dataset_type,
            model_name=self.model_name,
            img_size=train_img_size,
            batch_size=opt.batch_size,
            num_workers=opt.n_cpu,
        )

        self.train_dataloader = dataset_splits["train"]
        self.train_dataset = self.train_dataloader.dataset
        self.val_dataloader = dataset_splits["val"]
        self.num_classes = self.train_dataset.num_classes

        d = datetime.datetime.now()
        run_id = '{:%Y-%m-%d__%H-%M-%S}'.format(d)
        self.weight_path = (
            weight_path
            / self.model_name
            / "{}_{}_cls".format(opt.dataset_type, self.num_classes)
            / run_id
        )
        Path(self.weight_path).mkdir(parents=True, exist_ok=True)
        self.tb_writer = SummaryWriter(self.weight_path)

        self.model = create_model(
            model_name=self.model_name,
            pretraining_dataset=opt.pretraining_source_dataset,
            pretrained=opt.pretrained,
            num_classes=self.num_classes,
            progress=True,
            device=self.device,
        )

        self.criterion = YoloV5Loss(
            model=self.model,
            num_classes=self.num_classes,
            device=self.device,
            hyp_cfg=self.hyp_config,
        )
        if resume:
            self.__load_model_weights(weight_path, resume)

        self.optimizer, self.scheduler = make_od_optimizer(
            self.model,
            num_iters_per_epoch=len(self.train_dataloader),
            epochs=self.epochs,
            hyp_config=self.hyp_config,
            tb_writer=self.tb_writer
        )
        self.scaler = amp.GradScaler()

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

    def generate_file_hash(self, file_path, hash_size=16):
        with open(file_path, "rb") as f:
            bytes = f.read() # read entire file as bytes
            readable_hash = hashlib.sha256(bytes).hexdigest()
        return readable_hash[:hash_size]

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
            gt = SubsampledCOCO(
                opt.img_dir / "annotations/instances_val2017.json",
                subsample_categories=["car"],
            )

        Aps = eval_func(
            self.model,
            test_set,
            gt=gt,
            num_classes=self.num_classes,
            _set=opt.dataset_type,
            device=self.device,
            net=opt.net,
            img_size=opt.test_img_res,
            progressbar=True,
        )
        return Aps

    def train(self):
        print(self.model)
        print(
            "The number of samples in the train dataset split: {}".format(
                len(self.train_dataset)
            )
        )

        if opt.generate_checkpoint_name:
            sd = torch.load(opt.generate_checkpoint_name)
            self.model.load_state_dict(sd, strict=True)
            Aps = self.evaluate()
            model_name = opt.net.replace('yolo', 'yolov')
            dataset_name = f"{opt.dataset_type}-{self.num_classes}classes"
            map = str(math.ceil(1000 * Aps['mAP']) / 10).replace('.', '')[:3]
            hash = self.generate_file_hash(opt.generate_checkpoint_name)
            zoo_checkpoint_name = f"{model_name}-{dataset_name}-{map}-{hash}.pt"
            print(zoo_checkpoint_name)
            return

        if opt.eval_before_train:
            Aps = self.evaluate()
            print(f"Initial mAP values: {Aps}")

        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()

            loss_giou_mean = AverageMeter()
            loss_conf_mean = AverageMeter()
            loss_cls_mean = AverageMeter()
            loss_mean = AverageMeter()

            mloss = torch.zeros(4)
            self.optimizer.zero_grad()
            for i, (imgs, targets, labels_length, _) in enumerate(self.train_dataloader):
                self.scheduler.step()
                with amp.autocast():
                    imgs = imgs.to(self.device)
                    p, p_d = self.model(imgs)
                    loss, loss_giou, loss_conf, loss_cls = self.criterion(
                        p, p_d, targets, labels_length, imgs.shape[-1]
                    )
                    # Update running mean of tracked metrics
                    loss_items = torch.tensor([loss_giou, loss_conf, loss_cls, loss])
                    mloss = (mloss * i + loss_items) / (i + 1)

                    loss_giou_mean.update(loss_giou, imgs.size(0))
                    loss_conf_mean.update(loss_conf, imgs.size(0))
                    loss_cls_mean.update(loss_cls, imgs.size(0))
                    loss_mean.update(loss, imgs.size(0))

                    global_step = i + len(self.train_dataloader) * epoch
                    self.tb_writer.add_scalar('train/giou_loss', loss_giou_mean.avg, global_step)
                    self.tb_writer.add_scalar('train/conf_loss', loss_conf_mean.avg, global_step)
                    self.tb_writer.add_scalar('train/cls_loss', loss_cls_mean.avg, global_step)
                    self.tb_writer.add_scalar('train/loss', loss_mean.avg, global_step)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)  # optimizer.step
                self.scaler.update()
                self.optimizer.zero_grad()

                print(
                    f"\repoch {epoch}/{self.epochs} - Iteration: {i}/{len(self.train_dataloader)}, "
                    f" loss: giou {mloss[0]:0.4f}    conf {mloss[1]:0.4f}    cls {mloss[2]:0.4f}    "
                    f"loss {mloss[3]:0.4f}",
                    end="",
                )

                # multi-scale training (320-608 pixel resolution)
                if self.multi_scale_train:
                    self.train_dataset._img_size = random.choice(range(10, 20)) * 32

            mAP = 0
            if epoch % opt.eval_freq == 0:
                Aps = self.evaluate()
                mAP = Aps["mAP"]
                self.tb_writer.add_scalar('eval/mAP', mAP, epoch)
                self.__save_model_weights(epoch, mAP)
                print("best mAP : %g" % (self.best_mAP))


def make_od_optimizer(
    model, num_iters_per_epoch, epochs,
    hyp_config=None, hyp_config_name=None, tb_writer=None
):
    if hyp_config is None:
        hyp_config = HP_CONFIG_MAP.get(hyp_config_name, hyp_cfg_scratch)

    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(
            v.weight, nn.Parameter
        ):  # weight (with decay)
            g1.append(v.weight)
    optimizer = optim.SGD(
        g0,
        lr=hyp_config.TRAIN["lr0"],
        momentum=hyp_config.TRAIN["momentum"],
        nesterov=True,
    )
    optimizer.add_param_group(
        {"params": g1, "weight_decay": hyp_config.TRAIN["weight_decay"]}
    )  # add g1 with weight_decay
    optimizer.add_param_group({"params": g2})  # add g2 (biases)

    # Scheduler
    tb_write_fn = lambda lr, t: tb_writer.add_scalar("train/learning_rate", lr, t)
    scheduler = CosineDecayLR(
        optimizer,
        T_max=epochs * num_iters_per_epoch,
        lr_init=hyp_config.TRAIN["lr0"],
        lr_min=hyp_config.TRAIN["lr0"] * hyp_config.TRAIN["lrf"],
        warmup=hyp_config.TRAIN["warmup_epochs"] * num_iters_per_epoch,
        writer_step_fn=tb_write_fn,
    )

    return optimizer, scheduler


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img-dir",
        dest="img_dir",
        type=Path,
        help="The path to the folder containing images to be detected or trained.",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=10,
        help="The number of samples in one batch during training or inference.",
    )
    parser.add_argument(
        "--epochs",
        dest="epochs",
        type=int,
        default=False,
        help="The number of training epochs. If False, the default config value is used.",
    )
    parser.add_argument(
        "--eval-freq",
        dest="eval_freq",
        type=int,
        default=10,
        help="Evaluation run frequency (in training epochs).",
    )
    parser.add_argument(
        "--weight_path",
        type=Path,
        default="models",
        help="where weights should be stored",
    )
    parser.add_argument(
        "--checkpoint_save_freq",
        type=int,
        default=10,
        help="Checkpoint dump frequency in training epochs",
    )
    parser.add_argument(
        "--resume", action="store_true", default=False, help="Resume training flag"
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=False,
        help="Train the model from scratch if False",
    )
    parser.add_argument(
        "--pretraining_source_dataset",
        type=str,
        default="voc_20",
        help="Load pretrained weights fine-tuned on the specified dataset ('voc_20' or 'coco_80')",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument(
        "--n-cpu",
        dest="n_cpu",
        type=int,
        default=4,
        help="The number of cpu threads to use during batch generation.",
    )
    parser.add_argument(
        "--dataset",
        dest="dataset_type",
        type=str,
        default="voc",
        choices=[
            "coco",
            "voc",
            "lisa",
            "lisa_full",
            "lisa_subset11",
            "wider_face",
            "person_detection",
            "voc07",
            "car_detection",
            "person_pet_vehicle_detection",
        ],
        help="Name of the dataset to train/validate on",
    )
    parser.add_argument(
        "--net",
        dest="net",
        type=str,
        default="yolo5_6m",
        help="Specific YOLO model name to be used in training (ex. yolo3, yolo4m, yolo5_6s, ...)",
    )
    parser.add_argument(
        "--hp_config",
        dest="hp_config",
        type=str,
        default=None,
        help="The hyperparameter configuration name to use. Available options: 'scratch', 'finetune'",
    )
    parser.add_argument(
        "--test_img_res",
        dest="test_img_res",
        type=int,
        default=448,
        help="Image resolution to use during model testing",
    )
    parser.add_argument(
        "--train_img_res",
        dest="train_img_res",
        type=int,
        default=False,
        help="Image resolution to use during model training. If False, the default config value is used.",
    )
    parser.add_argument(
        "--eval_before_train",
        dest="eval_before_train",
        action="store_true",
        default=False,
        help="Run model evaluation before training",
    )
    parser.add_argument(
        "--evaluate",
        dest="evaluate",
        action="store_true",
        default=False,
        help="Run model evaluation only",
    )
    parser.add_argument(
        "--generate_checkpoint_name",
        dest="generate_checkpoint_name",
        type=str,
        default=False,
        help="Path to the checkpoint file to generate the DL torch zoo name for",
    )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    trainer = Trainer(weight_path=opt.weight_path, resume=opt.resume, gpu_id=opt.gpu_id)

    if opt.evaluate:
        Aps = trainer.evaluate()
        print(f"Evaluated mAP values: {Aps}")
    else:
        trainer.train()
