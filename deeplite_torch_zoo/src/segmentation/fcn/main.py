import warnings

import torch
from data_loader import get_loader
from trainer import Trainer
from utils import get_config, get_cuda, get_log_dir

warnings.filterwarnings("ignore")

resume = ""

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "val", "trainval", "demo"],
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--backbone", type=str, default="vgg")
    parser.add_argument("--root_dataset", type=str, default="data/VOC/")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument(
        "--fcn", type=str, default="32s", choices=["32s", "16s", "8s", "50", "101"]
    )
    opts = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpu_id)
    opts.cuda = get_cuda(torch.cuda.is_available() and opts.gpu_id != -1, opts.gpu_id)
    print("Cuda", opts.cuda)
    cfg = get_config()[1]
    opts.cfg = cfg

    if opts.mode in ["train", "trainval"]:
        opts.out = get_log_dir("fcn" + opts.fcn, 1, cfg)
        print("Output logs: ", opts.out)

    data = get_loader(opts)

    trainer = Trainer(data, opts)
    if opts.mode == "val":
        trainer.Test()
    elif opts.mode == "demo":
        trainer.Demo()
    else:
        trainer.Train()
