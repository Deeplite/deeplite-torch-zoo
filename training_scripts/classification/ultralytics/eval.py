# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

import argparse
import os
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import torch.nn as nn


import matplotlib.pyplot as plt


import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from deeplite_torch_zoo import (create_model, get_data_splits_by_name,
                                get_eval_function)
import numpy as np 
from torch.cuda import amp
from tqdm import tqdm

from kd import KDTeacher
from utils.general import (LOGGER, WorkingDirectory, colorstr, increment_path,
                           init_seeds, print_args, yaml_save)
from utils.torch_utils import (GenericLogger, ModelEMA, select_device,
                               smart_DDP, smart_optimizer,
                               smartCrossEntropyLoss,
                               torch_distributed_zero_first)

_DROPBLOCK = True  
if not _DROPBLOCK : 
    from deeplite_torch_zoo.src.classification.dropblock_models.resnet import resnet18 
else : 
    from deeplite_torch_zoo.src.classification.dropblock_models.resnet_dropblock import resnet18 


ROOT = Path.cwd()

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def train(opt, device):
    
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    save_dir, bs, epochs, nw, imgsz, pretrained = \
        opt.save_dir, opt.batch_size, opt.epochs, min(os.cpu_count() - 1, opt.workers), \
        opt.imgsz, str(opt.pretrained).lower() == 'true'
    cuda = device.type != 'cpu'

    base_dir = "/home/sudhakar/prj/all_neutrino/zoo/deeplite-torch-zoo/runs/train-cls/"
    # model_name = "food101_dropblock0.5_alter_epochs400" 
    # model_name = "food101_no_dropblock_epochs200"
    # model_name = "food101_dropblock_alternate_correct_epochs400"
    model_name = "food101_blocksparse_alter_epochs400"
    model_path = os.path.join(base_dir, model_name, "weights/last.pt")
    
    # Directories
    
    # Dataloaders
    dataloaders = get_data_splits_by_name(
        data_root=opt.data_root,
        dataset_name=opt.dataset,
        model_name=opt.model,
        batch_size=bs,
        test_batch_size=opt.test_batch_size,
        img_size=imgsz,
        num_workers=nw,
    )
    trainloader, testloader = dataloaders['train'], dataloaders['test']

    # DropBlock prob with polyDecay trend (TO-DO: Get Params from args)
    power = 2 # Exp of the function (if power = 1 >> linear)
    initial_value = 0.01
    final_value = 0.50
    begin_step = 5
    end_step = 388
    def get_dropProb(step):
        if step % 2 == 0 : 
            return 0.0
        drop_values = np.linspace(start=initial_value, stop=final_value, num=int(end_step-begin_step))
        drop_prob = drop_values[step]
        return drop_prob
        # p = min( 1.0, max(0.0,  (step - begin_step) / ((end_step - begin_step)), ),)
        # return final_value + (initial_value - final_value) * ((1 - p) ** power)
    
    with torch_distributed_zero_first(LOCAL_RANK), WorkingDirectory(ROOT):

        model = resnet18(pretrained=True)
        model.output = nn.Linear(in_features=512, out_features=101, bias=True) 
    
   
        
    model = model.to(device)
    # model.load_state_dict(torch.load(model_path))
    # model.load(model_path)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model'].state_dict())
    model.eval()   
    
    # Eval function
    evaluation_fn = get_eval_function(
        model_name=opt.model,
        dataset_name=opt.pretraining_dataset,
    )

    epochs = 20
    x_values = []
    y_values = [] 
    for epoch in range(10):  # loop over the dataset multiple times
        # dropblock_prob = epoch * 0.05
        dropblock_prob = 0.40
        print (f'evaluating epoch {epoch}...')
         ## DropBlock Update Prob
        for n, m in model.named_modules():
            if hasattr(m, 'dropblock'):
                m.update_dropProb(dropblock_prob)
        model.eval()
        metrics = evaluation_fn(model, testloader, progressbar=False)
        top1, top5 = metrics['acc'], metrics['acc_top5']
        print (f'dropblock_prob : {dropblock_prob:.2f}, top1 : {top1:.2f}, top5: {top5:.2f}')  
        
        
        x_values.append(float("{:.4f}".format(dropblock_prob))) 
        y_values.append(float("{:.4f}".format(top1))) 
    
    print (x_values)
    print (y_values)
    

    # dropblock 0.5 trained model
    # [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    # [0.8, 0.8, 0.8, 0.79, 0.79, 0.77, 0.76, 0.74, 0.72, 0.68] # db 0.5 trained
    # [0.82, 0.81, 0.81, 0.8, 0.78, 0.75, 0.7, 0.65, 0.57, 0.47] # db 0.25 trained 
    # [0.83, 0.63, 0.22, 0.05, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01] # no db 
    
    # [1, 2, 3,4,5,6,7,8,9,10]
    # [0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.63, 0.62, 0.62] # db - 0.5 
    # [0.6747, 0.674, 0.6756, 0.676, 0.6743, 0.679, 0.6756, 0.6789, 0.6765, 0.6728] # db - 0.45
    # [0.7158, 0.7136, 0.7147, 0.7163, 0.7138, 0.7167, 0.7141, 0.717, 0.715, 0.7144] # db - 0.40 
    # [0.7432, 0.7404, 0.7418, 0.7424, 0.741, 0.7441, 0.7407, 0.7465, 0.7438, 0.7427] # db - 0.35 
    # [0.7616, 0.7617, 0.7626, 0.7637, 0.7594, 0.763, 0.7617, 0.765, 0.7625, 0.7619] # db - 0.30 
    # [0.7746, 0.7762, 0.7752, 0.7796, 0.7776, 0.7753, 0.776, 0.7807, 0.7787, 0.7749] db - 0.25 
    # [0.787, 0.7859, 0.7855, 0.7877, 0.7864, 0.7876, 0.7864, 0.7886, 0.787, 0.7868] # db - 0.20  
    # [0.7975, 0.7968, 0.7952, 0.7975, 0.797, 0.7968, 0.7962, 0.7971, 0.7967, 0.7957] # db - 0.10 
    
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='./')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='flowers102', help='cifar10, cifar100, mnist, imagenet, ...')
    parser.add_argument('--pretraining-dataset', type=str, default='imagenet')
    parser.add_argument('--epochs', type=int, default=400, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='total batch size for all GPUs')
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

    parser.add_argument('--kd_model_name', default=None, type=str)
    parser.add_argument('--kd_model_checkpoint', default=None, type=str)
    parser.add_argument('--alpha_kd', default=5, type=float)
    parser.add_argument('--use_kd_only_loss', action='store_true', default=False)

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt):
    # Checks
    if RANK in {-1, 0}:
        print_args(vars(opt))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
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
    train(opt, device)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
