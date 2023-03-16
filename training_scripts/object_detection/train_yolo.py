# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

import argparse
import datetime
import logging
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from utils.general import (AverageMeter, colorstr, init_seeds, one_cycle,
                           print_args, set_logging, strip_optimizer)
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel,
                               select_device)

import deeplite_torch_zoo.src.objectdetection.yolov5.configs.hyps.hyp_config_default as hyp_cfg_scratch
import deeplite_torch_zoo.src.objectdetection.yolov5.configs.hyps.hyp_config_finetune as hyp_cfg_finetune
import deeplite_torch_zoo.src.objectdetection.yolov5.configs.hyps.hyp_config_lisa as hyp_cfg_lisa
from deeplite_torch_zoo import (create_model, get_data_splits_by_name,
                                get_eval_function)
from deeplite_torch_zoo.src.objectdetection.yolov5.models.losses.yolov5_loss import \
    YoloV5Loss
from deeplite_torch_zoo.src.objectdetection.yolov5.models.losses.yolox_loss import \
    ComputeXLoss

LOGGER = logging.getLogger(__name__)
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def get_hyperparameter_dict(dataset_name, hp_config=None):
    DATASET_TO_HP_CONFIG_MAP = {
        "lisa": hyp_cfg_lisa,
    }

    for dataset_name in (
        "voc",
        "coco",
        "wider_face",
        "person_detection",
        "car_detection",
        "voc07",
    ):
        DATASET_TO_HP_CONFIG_MAP[dataset_name] = hyp_cfg_scratch

    HP_CONFIG_MAP = {
        "scratch": hyp_cfg_scratch,
        "finetune": hyp_cfg_finetune,
    }

    if hp_config is None:
        for dataset_name in DATASET_TO_HP_CONFIG_MAP:
            if dataset_name in dataset_name:
                hyp_config = DATASET_TO_HP_CONFIG_MAP[dataset_name]
    else:
        hyp_config = HP_CONFIG_MAP[hp_config]
    return hyp_config.TRAIN, hyp_config


def train(opt, device):
    epochs, batch_size, noval, nosave, workers, freeze, = \
        opt.epochs, opt.batch_size, opt.noval, opt.nosave, opt.workers, opt.freeze

    d = datetime.datetime.now()
    run_id = '{:%Y-%m-%d__%H-%M-%S}'.format(d)
    save_dir = Path(opt.save_dir) / run_id

    # Directories
    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    # Get hyperparameter dict
    hyp, hyp_loss = get_hyperparameter_dict(opt.dataset_name, opt.hp_config)

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)
    tb_writer = SummaryWriter(save_dir)
    opt.img_dir = Path(opt.img_dir)

    # Config
    cuda = device.type != 'cpu'
    init_seeds(1 + RANK)

    # Dataloaders
    dataset_kwargs = {}
    if opt.img_size:
        dataset_kwargs = {'img_size': opt.img_size}
    dataset_splits = get_data_splits_by_name(
        data_root=opt.img_dir,
        dataset_name=opt.dataset_name,
        model_name=opt.model_name,
        batch_size=batch_size,
        num_workers=workers,
        distributed=(cuda and RANK != -1),
        **dataset_kwargs
    )
    train_img_size = dataset_splits["train"].dataset._img_size
    test_img_size = dataset_splits["test"].dataset._img_size

    train_loader = dataset_splits['train']
    test_loader = dataset_splits['test']

    dataset = train_loader.dataset
    nc = dataset.num_classes
    print("Number of classes = ", nc)

    nb = len(train_loader)  # number of batches

    # Model
    model = create_model(
        model_name=opt.model_name,
        pretraining_dataset=opt.pretraining_dataset,
        pretrained=opt.pretrained,
        num_classes=nc,
        device=device,
    )

    # Freeze
    freeze = [f'model.{x}.' for x in range(freeze)]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print(f'freezing {k}')
            v.requires_grad = False

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)

    if opt.adam:
        optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g2})  # add g2 (biases)
    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                f"{len(g0)} weight, {len(g1)} weight (no decay), {len(g2)} bias")
    del g0, g1, g2

    # Scheduler
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model) if RANK in [-1, 0] else None

    start_epoch, best_fitness = 0, 0.0

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)

    if hasattr(model, 'detection'):
        nl = model.detection.nl
    else:
        nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        logging.warning('DP not recommended, instead use torch.distributed.run for best DDP Multi-GPU results.\n'
                        'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    # Process 0
    if RANK in [-1, 0]:
        # Anchors
        model.half().float()  # pre-reduce anchor precision

    # DDP mode
    if cuda and RANK != -1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    # Model parameters
    hyp['giou'] *= 3. / nl  # scale to layers
    hyp['box'] = hyp['giou']
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (train_img_size / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model

    eval_function = get_eval_function(dataset_name=opt.dataset_name,
        model_name=opt.model_name)

    if 'yolox' in opt.model_name:
        criterion = ComputeXLoss(
            model=model,
            device=device,
        )
    else:
        criterion = YoloV5Loss(
            model=model,
            num_classes=nc,
            device=device,
            hyp_cfg=hyp_loss,
        )

    if opt.eval_before_train:
        ap_dict = eval_function(model, test_loader)
        LOGGER.info(f'Eval metrics: {ap_dict}')

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    last_opt_step = -1
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    stopper = EarlyStopping(patience=opt.patience)

    loss_giou_mean = AverageMeter()
    loss_conf_mean = AverageMeter()
    loss_cls_mean = AverageMeter()
    loss_mean = AverageMeter()

    ap_dict = {'mAP': 0}

    LOGGER.info(f'Image sizes {train_img_size} train, {test_img_size} val\n'
                f'Using {train_loader.num_workers} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')
    for epoch in range(start_epoch, epochs):  # epoch
        model.train()

        if not 'yolox' in opt.model_name:
            mloss = torch.zeros(3, device=device)  # mean losses
        else:
            mloss = torch.zeros(4, device=device)  # mean losses
        LOGGER.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))

        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        if RANK in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, labels_length, _) in pbar:  # batch
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float()

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(train_img_size * 0.5, train_img_size * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward

                loss, loss_items = criterion(
                    pred, targets, labels_length, imgs.shape[-1]
                )

                if RANK in (-1, 0):
                    loss_giou_mean.update(loss_items[0], imgs.size(0))
                    loss_conf_mean.update(loss_items[1], imgs.size(0))
                    loss_cls_mean.update(loss_items[2], imgs.size(0))
                    loss_mean.update(loss, imgs.size(0))

                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log
            if RANK in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%10s' * 2 + '%10.4g' * 5) % (
                    f'{epoch}/{epochs - 1}', mem, *mloss[:3], targets.shape[0], imgs.shape[-1]))
            # end batch

        # Scheduler
        scheduler.step()

        if RANK in [-1, 0]:
            for idx, param_group in enumerate(optimizer.param_groups):
                tb_writer.add_scalar(f'learning_rate/gr{idx}', param_group['lr'], epoch)
            tb_writer.add_scalar('train/giou_loss', loss_giou_mean.avg, epoch)
            tb_writer.add_scalar('train/conf_loss', loss_conf_mean.avg, epoch)
            tb_writer.add_scalar('train/cls_loss', loss_cls_mean.avg, epoch)
            tb_writer.add_scalar('train/loss', loss_mean.avg, epoch)

            # mAP
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if (not noval or final_epoch) and epoch > opt.eval_skip_epochs - 1:  # Calculate mAP
                ap_dict = eval_function(ema.ema, test_loader)
                LOGGER.info(f'Eval metrics: {ap_dict}')
                tb_writer.add_scalar('eval/mAP', ap_dict['mAP'], epoch)
                for eval_key, eval_value in ap_dict.items():
                    if eval_key != 'mAP':
                        tb_writer.add_scalar(f'ap_per_class/{eval_key}', eval_value, epoch)

            # Update best mAP
            fi = ap_dict['mAP']
            if fi > best_fitness:
                best_fitness = fi

            # Save model
            if (not nosave) or final_epoch:  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'model': deepcopy(de_parallel(model)).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if (epoch > 0) and (opt.save_period > 0) and (epoch % opt.save_period == 0):
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt

            # Stop Single-GPU
            if RANK == -1 and stopper(epoch=epoch, fitness=fi):
                break

        # end epoch
    # end training
    if RANK in [-1, 0]:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f'\nValidating {f}...')
                    ckpt = torch.load(f, map_location=device)
                    model = ckpt['ema' if ckpt.get('ema') else 'model']
                    model.float().eval()
                    ap_dict = eval_function(model, test_loader)
                    LOGGER.info(f'Eval metrics: {ap_dict}')

        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

    torch.cuda.empty_cache()


def parse_opt(known=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--img-dir', dest='img_dir', type=str,
        help='the path to the folder containing images to be detected or trained')
    parser.add_argument('--pretrained', action='store_true', default=False,
        help='train the model from scratch if false')
    parser.add_argument(
        "--pretraining_dataset", type=str, default="coco",
        help="Load pretrained weights fine-tuned on the specified dataset ('voc' or 'coco')",
    )
    parser.add_argument(
        "--dataset", dest="dataset_name", type=str, default="voc",
        choices=[
            "coco",
            "voc",
            "lisa",
            "lisa_full",
            "lisa_subset11",
            "wider_face",
            "voc07",
            "car_detection",
            "voc_format_dataset",
        ],
        help="Name of the dataset to train/validate on",
    )
    parser.add_argument(
        "--model", dest="model_name", type=str, default="yolo5n",
        help="Specific YOLO model name to be used in training (ex. yolo3, yolo4m, yolo5n, ...)",
    )
    parser.add_argument(
        "--hp_config",
        dest="hp_config", type=str, default=None,
        help="The hyperparameter configuration name to use. Available options: 'scratch', 'finetune'",
    )
    parser.add_argument(
        "--img_size",
        dest="img_size", type=int, default=False,
        help="Image resolution to use during model training. If False, the default config value is used.",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        dest="save_dir",
        default="runs",
        help="Log directory",
    )
    parser.add_argument(
        "--eval-skip-epochs",
        dest="eval_skip_epochs",
        type=int,
        default=100,
        help="Skip evaluation for this number of epochs in the beginning",
    )

    parser.add_argument('--eval_before_train', action='store_true', help='run eval before training starts')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=64, help='total batch size for all GPUs')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze. backbone=10, all=24')
    parser.add_argument('--checkpoint_save_freq', dest='save_period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt):
    # Checks
    set_logging(RANK)
    if RANK in [-1, 0]:
        print_args(vars(opt))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        assert opt.batch_size % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    train(opt, device)
    if WORLD_SIZE > 1 and RANK == 0:
        LOGGER.info('Destroying process group... ')
        dist.destroy_process_group()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
