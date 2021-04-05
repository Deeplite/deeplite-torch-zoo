import argparse
import logging
import os
import sys
from pathlib import Path

import albumentations as albu
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from deeplite_torch_zoo.segmentation.datasets.carvana import BasicDataset
from deeplite_torch_zoo.segmentation.datasets.pascal_voc import \
    PascalVocDataset as Dataset
from deeplite_torch_zoo.src.segmentation.Unet.eval import eval_net
from deeplite_torch_zoo.src.segmentation.Unet.model.unet_model import UNet

data_root = "/neutrino/datasets/carvana/"
dir_checkpoint = "checkpoints/"


def train_net(
    net,
    device,
    epochs=51,
    batch_size=1,
    lr=0.001,
    val_percent=0.1,
    save_cp=True,
    img_scale=0.5,
):

    if False:
        # Dataset
        affine_augmenter = albu.Compose(
            [
                albu.HorizontalFlip(p=0.5),
                # Rotate(5, p=.5)
            ]
        )
        # image_augmenter = albu.Compose([albu.GaussNoise(p=.5),
        #                                 albu.RandomBrightnessContrast(p=.5)])
        image_augmenter = None
        data_dir = "deeplite_torch_zoo/data/VOC/VOCdevkit/VOC2012/"
        train_dataset = Dataset(
            base_dir=data_dir,
            split="train",
            affine_augmenter=affine_augmenter,
            image_augmenter=image_augmenter,
            net_type="unet",
        )
        valid_dataset = Dataset(base_dir=data_dir, split="valid", net_type="unet")
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            valid_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
        )

        n_train = len(train_dataset)
        n_val = len(valid_dataset)
    else:
        train_dataset = BasicDataset(
            os.path.join(data_root, "train_imgs/"),
            os.path.join(data_root, "train_masks/"),
            img_scale,
        )
        valid_dataset = BasicDataset(
            os.path.join(data_root, "val_imgs/"),
            os.path.join(data_root, "val_masks/"),
            img_scale,
        )
        n_val = len(valid_dataset)
        n_train = len(train_dataset)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )

    global_step = 0

    logging.info(
        """Starting training:
        Epochs:          {}
        Batch size:      {}
        Learning rate:   {}
        Training size:   {}
        Validation size: {}
        Checkpoints:     {}
        Device:          {}
        Images scaling:  {}
    """.format(
            epochs, batch_size, lr, n_train, n_val, save_cp, device.type, img_scale
        )
    )

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min" if net.n_classes > 1 else "max", patience=10
    )
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    best_valid = 1e6
    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(
            total=n_train, desc="Epoch {}/{}".format(epoch + 1, epochs), unit="img"
        ) as pbar:
            for batch in train_loader:
                imgs = batch[0]
                true_masks = batch[1]  # .unsqueeze(0)
                assert imgs.shape[1] == net.n_channels, (
                    "Network has been defined with {} input channels, "
                    "but loaded images have {} channels. Please check that "
                    "the images are loaded correctly.".format(
                        net.n_channels, imgs.shape[1]
                    )
                )

                imgs = imgs.to(device=device)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()

                pbar.set_postfix(**{"loss (batch)": loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

            # if epoch % 10 == 0:
            for tag, value in net.named_parameters():
                tag = tag.replace(".", "/")
            val_score = eval_net(net, val_loader, device)
            scheduler.step(val_score)

            if net.n_classes > 1:
                logging.info("Validation cross entropy: {}".format(val_score))
            else:
                logging.info("Validation Dice Coeff: {}".format(val_score))

        if save_cp:
            dir_checkpoint = Path(
                "deeplite_torch_zoo/weight/segmentation/UNet_carvana/"
            )
            try:
                dir_checkpoint.mkdir(parents=True, exist_ok=True)
                logging.info("Created checkpoint directory")
            except OSError:
                pass
            torch.save(
                net.state_dict(),
                dir_checkpoint / "CP_epoch{epoch}.pth".format(epoch=epoch + 1),
            )
            logging.info("Checkpoint {epoch} saved !".format(epoch=epoch + 1))


def get_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-e",
        "--epochs",
        metavar="E",
        type=int,
        default=5,
        help="Number of epochs",
        dest="epochs",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        metavar="B",
        type=int,
        nargs="?",
        default=2,
        help="Batch size",
        dest="batchsize",
    )
    parser.add_argument(
        "-l",
        "--learning-rate",
        metavar="LR",
        type=float,
        nargs="?",
        default=0.0001,
        help="Learning rate",
        dest="lr",
    )
    parser.add_argument(
        "-f",
        "--load",
        dest="load",
        type=str,
        default=None,
        help="Load model from a .pth file",
    )
    parser.add_argument(
        "-s",
        "--scale",
        dest="scale",
        type=float,
        default=0.5,
        help="Downscaling factor of the images",
    )
    parser.add_argument(
        "-v",
        "--validation",
        dest="val",
        type=float,
        default=10.0,
        help="Percent of the data that is used as validation (0-100)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device {}".format(device))

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=3, n_classes=1, bilinear=True)
    logging.info(
        "Network:\n"
        "\t{} input channels\n"
        "\t{} output channels (classes)\n"
        "\t{} upscaling".format(
            net.n_channels,
            net.n_classes,
            "Bilinear" if net.bilinear else "Transposed conv",
        )
    )

    if False:
        pretrained_net = torch.hub.load("milesial/Pytorch-UNet", "unet_carvana")
        net.load_state_dict(pretrained_net.state_dict())

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info("Model loaded from {}".format(args.load))

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(
            net=net,
            epochs=args.epochs,
            batch_size=args.batchsize,
            lr=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
        )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), "INTERRUPTED.pth")
        logging.info("Saved interrupt")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
