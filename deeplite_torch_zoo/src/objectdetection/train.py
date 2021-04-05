import argparse
import os
import random
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm

import deeplite_torch_zoo.src.objectdetection.configs.coco_config as coco_cfg
import deeplite_torch_zoo.src.objectdetection.configs.hyp_config as hyp_cfg_scratch
import deeplite_torch_zoo.src.objectdetection.configs.hyp_finetune as hyp_cfg_finetune
import deeplite_torch_zoo.src.objectdetection.configs.lisa_config as lisa_cfg
import deeplite_torch_zoo.src.objectdetection.configs.voc_config as voc_cfg
import deeplite_torch_zoo.src.objectdetection.yolov3.utils.gpu as gpu
from deeplite_torch_zoo.wrappers.wrapper import get_data_splits_by_name
from deeplite_torch_zoo.wrappers.models import (yolo3, yolo4, yolo4_lisa, yolo5,
                                            yolo_eval_func)
from deeplite_torch_zoo.src.objectdetection.eval.lisa_eval import LISAEval
from deeplite_torch_zoo.src.objectdetection.yolov3.model.loss.yolo_loss import \
    YoloV3Loss
from deeplite_torch_zoo.src.objectdetection.yolov3.utils.cosine_lr_scheduler import \
    CosineDecayLR
from deeplite_torch_zoo.src.objectdetection.yolov3.utils.tools import (
    init_seeds, weights_init_normal)
from deeplite_torch_zoo.src.objectdetection.yolov5.models.yolov5_loss import \
    YoloV5Loss


class Trainer(object):
    def __init__(self, weight_path, resume, gpu_id):
        init_seeds(0)
        assert opt.n_cpu == 0, "multi-sclae need to be fixed if you must use multi cpus"

        assert opt.dataset_type in ["coco", "voc", "lisa", "lisa_full", "lisa_subset11"]
        assert opt.net in [
            "yolov3",
            "yolov5s",
            "yolov5m",
            "yolov5l",
            "yolov5x",
            "yolov4s",
            "yolov4m",
            "yolov4l",
            "yolov4x",
        ]
        if "yolov4" in opt.net or "yolov5" in opt.net:
            assert opt.net == opt.arch_cfg
        if opt.dataset_type == "voc":
            self.cfg = voc_cfg
        elif opt.dataset_type == "coco":
            self.cfg = coco_cfg
        elif "lisa" in opt.dataset_type:
            self.cfg = lisa_cfg

        # self.hyp_config = hyp_cfg_finetune if opt.pretrained else hyp_cfg_scratch
        self.hyp_config = hyp_cfg_scratch

        self.device = gpu.select_device(gpu_id, force_cpu=False)
        self.start_epoch = 0
        self.best_mAP = 0.0
        self.epochs = self.hyp_config.TRAIN["EPOCHS"]

        self.multi_scale_train = self.hyp_config.TRAIN["MULTI_SCALE_TRAIN"]

        dataset_splits = get_data_splits_by_name(
            data_root=opt.img_dir,
            dataset_name=opt.dataset_type,
            model_name=opt.net,
            batch_size=opt.batch_size,
            num_workers=opt.n_cpu,
            num_classes=self.cf.DATA["NUM"],
            img_size=self.hyp_config.TRAIN["TRAIN_IMG_SIZE"]
        )

        self.train_dataloader = dataset_splits["train"]
        self.train_dataset = self.train_dataloader.dataset
        self.val_dataloader = dataset_splits["val"]
        self.num_classes = self.train_dataset.num_classes
        self.weight_path = (
            weight_path
            / opt.net
            / "{}_{}_cls".format(opt.dataset_type, self.num_classes)
        )
        Path(self.weight_path).mkdir(parents=True, exist_ok=True)

        # self.yolov3 = yolo3().to(self.device)
        # self.yolov3 = yolo3(pretrained=opt.pretrained, progress=True, dataset=opt.dataset_type).to(self.device)
        # self.yolov3.apply(weights_init_normal)
        self.model = self._get_model()

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.hyp_config.TRAIN["LR_INIT"],
            momentum=self.hyp_config.TRAIN["MOMENTUM"],
            weight_decay=self.hyp_config.TRAIN["WEIGHT_DECAY"],
        )
        # self.optimizer = optim.Adam(self.yolov3.parameters(), lr = lr_init, weight_decay=0.9995)

        self.criterion = self._get_loss()
        if resume:
            self.__load_model_weights(weight_path, resume)

        self.scheduler = CosineDecayLR(
            self.optimizer,
            T_max=self.epochs * len(self.train_dataloader),
            lr_init=self.hyp_config.TRAIN["LR_INIT"],
            lr_min=self.hyp_config.TRAIN["LR_END"],
            warmup=self.hyp_config.TRAIN["WARMUP_EPOCHS"] * len(self.train_dataloader),
        )

    def _get_model(self):
        if "yolov3" in opt.net:
            return yolo3(
                pretrained=opt.pretrained,
                progress=True,
                num_classes=self.num_classes,
                device=self.device,
            )
        elif "yolov5" in opt.net:
            return yolo5(
                pretrained=opt.pretrained,
                num_classes=self.num_classes,
                net=opt.arch_cfg,
                device=self.device,
            )
        elif "yolov4" in opt.net:
            if "lisa" in opt.dataset_type:
                return yolo4_lisa(
                    pretrained=opt.pretrained,
                    num_classes=self.num_classes,
                    net=opt.arch_cfg,
                    device=self.device,
                )
            return yolo4(
                pretrained=opt.pretrained,
                num_classes=self.num_classes,
                net=opt.arch_cfg,
                device=self.device,
            )

    def _get_loss(self):
        return YoloV3Loss(num_classes=self.num_classes, device=self.device)
        if opt.net == "yolov3":
            return YoloV3Loss(num_classes=self.num_classes, device=self.device)
        else:
            return YoloV5Loss(
                model=self.model, num_classes=self.num_classes, device=self.device
            )

    def __load_model_weights(self, weight_path, resume):
        if resume:
            # last_weight = os.path.join(os.path.split(weight_path)[0], "last.pt")
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
        # best_weight = os.path.join(os.path.split(self.weight_path)[0], "best.pt")
        # last_weight = os.path.join(os.path.split(self.weight_path)[0], "last.pt")
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

        if epoch > 0 and epoch % 10 == 0:
            torch.save(
                chkpt,
                os.path.join(
                    os.path.split(self.weight_path)[0], "backup_epoch%g.pt" % epoch
                ),
            )
        del chkpt

    def train(self):
        print(self.model)
        print("Train datasets number is : {}".format(len(self.train_dataset)))
        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()

            mloss = torch.zeros(4)
            for i, (imgs, targets, labels_length, _) in tqdm(
                enumerate(self.train_dataloader)
            ):
                self.scheduler.step()
                imgs = imgs.to(self.device)

                # p, p_d = self.model(imgs)
                p, p_d = self.model(imgs)
                loss, loss_giou, loss_conf, loss_cls = self.criterion(
                    p, p_d, targets, labels_length, self.train_dataset._img_size
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update running mean of tracked metrics
                loss_items = torch.tensor([loss_giou, loss_conf, loss_cls, loss])
                mloss = (mloss * i + loss_items) / (i + 1)

                # Print batch results
                if (i + 1) % opt.print_freq == 0:
                    s = (
                        "Epoch:[ %d | %d ]    Batch:[ %d | %d ]    loss_giou: %.4f    loss_conf: %.4f    loss_cls: %.4f    loss: %.4f    "
                        "lr: %g"
                    ) % (
                        epoch,
                        self.epochs - 1,
                        i,
                        len(self.train_dataloader) - 1,
                        mloss[0],
                        mloss[1],
                        mloss[2],
                        mloss[3],
                        self.optimizer.param_groups[0]["lr"],
                    )
                    print(s)

                # multi-sclae training (320-608 pixels) every 10 batches
                if self.multi_scale_train and (i + 1) % 10 == 0:
                    self.train_dataset._img_size = random.choice(range(10, 20)) * 32
                    # print("multi_scale_img_size : {}".format(self.train_dataset._img_size))
            # mAP = 0
            # if epoch >= 20:
            #     print('*' * 20 + "Validate" + '*' * 20)
            #     with torch.no_grad():
            #         APs = Evaluator(self.yolov3).APs_voc()
            #         for i in APs:
            #             print("{} --> mAP : {}".format(i, APs[i]))
            #             mAP += APs[i]
            #         mAP = mAP / self.num_classes
            #         print('mAP:%g' % (mAP))
            mAP = 0
            if epoch % opt.eval_freq == 0:
                if opt.dataset_type in ["voc", "coco"]:
                    if opt.dataset_type == "voc":
                        test_set = opt.img_dir / "VOC2007"
                    elif opt.dataset_type == "coco":
                        test_set = opt.img_dir
                    Aps = yolo_eval_func(
                        self.model,
                        test_set,
                        num_classes=self.num_classes,
                        _set=opt.dataset_type,
                        device=self.device,
                        net=opt.net,
                    )
                    mAP = Aps["mAP"]
                elif "lisa" in opt.dataset_type:
                    mAP = LISAEval(self.model, opt.img_dir).evaluate()

                self.__save_model_weights(epoch, mAP)
                print("best mAP : %g" % (self.best_mAP))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img-dir",
        dest="img_dir",
        type=Path,
        default="data/VOC/VOCdevkit",  #'data/LISA', #'data/coco', #
        help="The path to the folder containing images to be detected or trained.",
    )
    parser.add_argument(
        "--arch-cfg",
        dest="arch_cfg",
        type=str,
        default="yolov4m",
        help="The path to yolo architecture file (only for yolo 4 and 5).",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=10,
        help="The number of sample in one batch during training or inference.",
    )
    parser.add_argument(
        "--print-freq",
        dest="print_freq",
        type=int,
        default=500,
        help="The number of sample in one batch during training or inference.",
    )
    parser.add_argument(
        "--eval-freq",
        dest="eval_freq",
        type=int,
        default=10,
        help="The number of sample in one batch during training or inference.",
    )
    parser.add_argument(
        "--weight_path",
        type=Path,
        default="deeplite_torch_zoo/weight/",
        help="where weights should be stored",
    )
    parser.add_argument(
        "--resume", action="store_false", default=False, help="resume training flag"
    )
    parser.add_argument(
        "--pretrained", default=True, help="Train Model from scratch if False"
    )

    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument(
        "--n-cpu",
        dest="n_cpu",
        type=int,
        default=0,
        help="The number of cpu thread to use during batch generation.",
    )
    parser.add_argument(
        "--dataset",
        dest="dataset_type",
        type=str,
        default="voc",
        help="The type of the dataset used. Currently support 'coco', 'voc', 'lisa', 'lisa_subset11' and 'image_folder'",
    )
    parser.add_argument(
        "--net",
        dest="net",
        type=str,
        default="yolov4m",
        help="The type of the network used. Currently support 'yolov3' and'yolov5'",
    )
    opt = parser.parse_args()

    Trainer(weight_path=opt.weight_path, resume=opt.resume, gpu_id=opt.gpu_id).train()
