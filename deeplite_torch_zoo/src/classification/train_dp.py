import argparse
import time, copy, sys, os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from deeplite_torch_zoo.wrappers import get_data_splits_by_name
from mmcv.runner import init_dist
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def cleanup():
    dist.destroy_process_group()


def _run_epoch(rank, world_size, epoch, phase="train", output_path=None, model=None, dataloaders=None, criterion=None, optimizer=None):

    if phase == 'train':
        model.train()
    else:
        model.eval()

    running_loss = AverageMeter()
    running_corrects = AverageMeter()

    # Iterate over data.
    for i, (inputs, labels) in enumerate(dataloaders[phase]):
        inputs.to(rank)
        labels.to(rank)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs.cuda(non_blocking=True))
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs.to(rank), labels.to(rank))

            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                optimizer.step()

        # statistics
        _loss = loss.item()
        _corrects = torch.sum(preds.to(rank) == labels.data.to(rank))
        _batch_size = inputs.size(0)
        running_loss.update(_loss, _batch_size)
        running_corrects.update(_corrects, _batch_size)
        print("\rIteration: {}/{}, Loss: {}.".format(i*world_size+rank+1, len(dataloaders[phase])*world_size, loss.item() * inputs.size(0)), end="")
        sys.stdout.flush()

    epoch_loss = running_loss.avg
    epoch_acc = running_corrects.avg
    if rank == 0:
        print(f'\n{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print()
        save_path = str(output_path) + '/model_{}_epoch.pt'.format(epoch+1)
        if phase == "train":
            torch.save(model.state_dict(), save_path)
            print(f"\nSaved: {save_path}")
    return epoch_loss, epoch_acc


def run_parallel(train_fn, world_size, output_path=None, model=None, criterion=None, optimizer=None, opt=None):
    mp.spawn(
        train_fn,
        args=(world_size, output_path, model, criterion, optimizer, opt),
        nprocs=world_size,
        join=True
    )


def train_model(rank, world_size, output_path=None, model=None, criterion=None, optimizer=None, opt=None):
    num_epochs = opt.epochs

    if not os.path.exists('models/' + str(output_path)):
        os.makedirs('models/' + str(output_path))
    since = time.time()

    dataloaders = get_data_splits_by_name(
        dataset_name=opt.dataset_name,
        data_root=opt.data_root,
        num_workers=0,
        batch_size=128,
        device="cuda",
        distributed=True
    )
    criterion = criterion.cuda(rank)
    model = DDP(model.to(rank), device_ids=[rank], output_device=rank)
    writer = SummaryWriter(f"{output_path}/tensorboard/")
    for epoch in range(num_epochs):
        if rank == 0:
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            loss, acc = _run_epoch(rank, world_size, epoch, phase=phase, output_path=output_path, model=model, dataloaders=dataloaders, criterion=criterion, optimizer=optimizer)
            if phase == "test":
                writer.add_scalar(f'Accuracy/test - N GPUS {world_size}', acc, epoch)
            elif phase == "train":
                writer.add_scalar(f'Accuracy/train - N GPUS {world_size}', acc, epoch)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    writer.close()
    cleanup()


def get_model(arch="resnet18", num_classes=100, pretrained=True, device="cuda"):
    if not pretrained:
        model = eval(f"models.{arch}")(num_classes=num_classes)
    else:
        weights = eval(f"models.{arch}")(pretrained=pretrained).state_dict()
        weights = {k: v for k, v in weights.items() if 'fc' not in k and 'classifier' not in k}
        model = eval(f"models.{arch}")(num_classes=num_classes)
        model.load_state_dict(weights, strict=False)
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="resnet18")
    parser.add_argument("--data-root", type=str, default="/neutrino/datasets/TinyImageNet/")
    parser.add_argument("--dataset-name", type=str, default="tinyimagenet")
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--pretrained", type=bool, default=True)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=8)

    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='pytorch',
        help='job launcher')
    opt = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(opt.local_rank)

    dist_params = dict(backend='nccl')
    init_dist(opt.launcher, **dist_params)
    model = get_model(arch=opt.arch, num_classes=opt.num_classes, pretrained=opt.pretrained, device="cuda")

    #Loss Function
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.001 * opt.world_size, momentum=0.9)
    train_model(opt.local_rank, opt.world_size, f"./models/{opt.dataset_name}/{opt.arch}", model, criterion, optimizer, opt=opt)
