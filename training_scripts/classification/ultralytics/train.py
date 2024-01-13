# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

import argparse
import os
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from torch.cuda import amp
from tqdm import tqdm

from deeplite_torch_zoo.utils import (LOGGER, GenericLogger, ModelEMA, colorstr, init_seeds, print_args,
                                      yaml_save, increment_path, WorkingDirectory,
                                      select_device, smart_DDP, smart_optimizer,
                                      smartCrossEntropyLoss, torch_distributed_zero_first)

from deeplite_torch_zoo import get_model, get_dataloaders, get_eval_function
from deeplite_torch_zoo.utils.kd import KDTeacher, compute_kd_loss

from torch.optim.lr_scheduler import _LRScheduler


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


ROOT = Path.cwd()

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def train(opt, model, device):

    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    save_dir, bs, epochs, nw, imgsz, pretrained = \
        opt.save_dir, opt.batch_size, opt.epochs, min(os.cpu_count() - 1, opt.workers), \
        opt.imgsz, str(opt.pretrained).lower() == 'true'
    cuda = device.type != 'cpu'

    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir

    # Save run settings
    yaml_save(save_dir / 'opt.yaml', vars(opt))

    # Dataloaders
    dataloaders = get_dataloaders(
        data_root=opt.data_root,
        dataset_name=opt.dataset,
        batch_size=bs,
        test_batch_size=opt.test_batch_size,
        img_size=imgsz,
        num_workers=nw,
    )
    trainloader, testloader = dataloaders['train'], dataloaders['test']

    # Model
    opt.num_classes = len(trainloader.dataset.classes)
    with torch_distributed_zero_first(LOCAL_RANK), WorkingDirectory(ROOT):
        model_kd = None
        if opt.kd_model_name is not None:
            model_kd = KDTeacher(opt)

    for p in model.parameters():
        p.requires_grad = True  # for training
    model = model.to(device)

    # Eval function
    evaluation_fn = get_eval_function(
        model_name=opt.model,
        dataset_name=opt.pretraining_dataset,
    )

    # Info
    if RANK in {-1, 0}:
        if opt.verbose:
            LOGGER.info(model)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # Scheduler
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[150, 225], gamma=0.1)
    warmup_scheduler = WarmUpLR(optimizer, len(trainloader))
    eval_from = 220

    # EMA
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # DDP mode
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # Train
    t0 = time.time()
    criterion = smartCrossEntropyLoss(label_smoothing=opt.label_smoothing)  # loss function
    best_fitness = 0.0
    best_top5 = 0.0
    scaler = amp.GradScaler(enabled=cuda)
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} test\n'
                f'Using {nw * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting {opt.model} training on {opt.dataset} dataset for {epochs} epochs...\n\n'
                f"{'Epoch':>10}{'GPU_mem':>10}{'train_loss':>12}{'top1_acc':>12}{'top5_acc':>12}")

    for epoch in range(epochs):  # loop over the dataset multiple times
        tloss, vloss, fitness = 0.0, 0.0, 0.0  # train loss, val loss, fitness
        model.train()
        pbar = enumerate(trainloader)
        if RANK in {-1, 0}:
            pbar = tqdm(enumerate(trainloader), total=len(trainloader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for i, (images, labels) in pbar:  # progress bar
            images, labels = images.to(device, non_blocking=True), labels.to(device)

            # Forward
            with amp.autocast(enabled=cuda):  # stability issues when enabled
                output = model(images)
                loss = criterion(output, labels)

                if model_kd is not None:
                    kd_loss = compute_kd_loss(images, output, model_kd, model)
                    # adding KD loss
                    if not opt.use_kd_loss_only:
                        loss += opt.alpha_kd * kd_loss
                    else:
                        loss = opt.alpha_kd * kd_loss

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            scaler.unscale_(optimizer)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema:
                ema.update(model)

            if epoch <= (opt.warmup_epochs - 1):
                warmup_scheduler.step()

            if RANK in {-1, 0}:
                # Print
                tloss = (tloss * i + loss.item()) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                pbar.desc = f"{f'{epoch + 1}/{epochs}':>10}{mem:>10}{tloss:>12.3g}" + ' ' * 36

                # Test
                if (epoch >= eval_from) and (i == len(pbar) - 1):  # last batch
                    metrics = evaluation_fn(
                        ema.ema,
                        testloader,
                        progressbar=False,
                        break_iter=None if not opt.dryrun else 1
                    )
                    top1, top5 = metrics['acc'], metrics['acc_top5']
                    fitness = top1  # define fitness as top1 accuracy
                    pbar.desc = f"{pbar.desc[:-36]}{top1:>12.3g}{top5:>12.3g}"

            if opt.dryrun:
                break

        # Scheduler
        if epoch > (opt.warmup_epochs - 1):
            scheduler.step()

        if opt.dryrun:
            break

        # Log metrics
        if RANK in {-1, 0}:
            # Best fitness
            if fitness > best_fitness:
                best_fitness = fitness
                best_top5 = top5

            # Log
            # metrics = {
            #     "train/loss": tloss,
            #     "metrics/accuracy_top1": top1,
            #     "metrics/accuracy_top5": top5,
            #     "lr/0": optimizer.param_groups[0]['lr']}  # learning rate
            # logger.log_metrics(metrics, epoch)

            # Save model
            final_epoch = epoch + 1 == epochs

            # if (not opt.nosave) or final_epoch:
            #     ckpt = {
            #         'epoch': epoch,
            #         'best_fitness': best_fitness,
            #         'model': deepcopy(ema.ema).half(),  # deepcopy(de_parallel(model)).half(),
            #         'ema': None,  # deepcopy(ema.ema).half(),
            #         'updates': ema.updates,
            #         'optimizer': None,  # optimizer.state_dict(),
            #         'opt': vars(opt),
            #         'date': datetime.now().isoformat()}

            #     # Save last, best and delete
            #     torch.save(ckpt, last)
            #     if best_fitness == fitness:
            #         torch.save(ckpt, best)
            #         torch.save(ckpt['model'].state_dict(), best_sd)
            #     del ckpt

    # Train complete
    if not opt.dryrun and RANK in {-1, 0} and final_epoch:
        return {'top1': best_fitness, 'top5': best_top5}

        # LOGGER.info(f'\nTraining complete ({(time.time() - t0) / 3600:.3f} hours)'
        #             f"\nResults saved to {colorstr('bold', save_dir)}")

        # Log results
        # meta = {"epochs": epochs, "top1_acc": best_fitness, "date": datetime.now().isoformat()}
        # logger.log_model(best, epochs, metadata=meta)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='./')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='cifar100', help='cifar10, cifar100, mnist, imagenet, ...')
    parser.add_argument('--pretraining-dataset', type=str, default='imagenet')
    parser.add_argument('--epochs', type=int, default=300, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--test-batch-size', type=int, default=256, help='testing batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=224, help='train, val image size (pixels)')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train-cls', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--optimizer', choices=['SGD', 'Adam', 'AdamW', 'RMSProp'], default='Adam', help='optimizer')
    parser.add_argument('--lr0', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--decay', type=float, default=5e-5, help='weight decay')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing epsilon')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
    parser.add_argument('--warmup_epochs', type=int, default=1, help='Warmup epochs')

    parser.add_argument('--kd_model_name', default=None, type=str)
    parser.add_argument('--kd_model_checkpoint', default=None, type=str)
    parser.add_argument('--alpha_kd', default=5, type=float)
    parser.add_argument('--use_kd_loss_only', action='store_true', default=False)

    parser.add_argument('--dryrun', action='store_true', help='Dry run mode for testing')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, model):
    # Checks
    if RANK in {-1, 0}:
        print_args(vars(opt))

    # DDP mode
    device = select_device(opt.device, batch=opt.batch_size)
    if LOCAL_RANK != -1:
        assert opt.batch_size != -1, 'AutoBatch is coming soon for classification, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Parameters
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run

    # Train
    return train(opt, model, device)
