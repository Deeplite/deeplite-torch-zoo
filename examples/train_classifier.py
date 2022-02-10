import argparse

import torch
import torch.nn.functional as F
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR

from deeplite_torch_zoo import get_data_splits_by_name, get_model_by_name


def train(args, model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch training Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--dataset", metavar="DATASET", default="cifar100", help="dataset to use"
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        metavar="N",
        default=4,
        help="number of data loading workers",
    )
    parser.add_argument(
        "-r", "--data_root", metavar="PATH", default="", help="dataset data root path"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument('-a', '--arch', metavar='ARCH', default='vgg19', help='model architecture')
    args = parser.parse_args()

    device = torch.device("cuda")

    data_splits = get_data_splits_by_name(
        dataset_name=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_torch_workers=args.workers,
    )
    model = get_model_by_name(
        model_name=args.arch,
        dataset_name=args.dataset,
        pretrained=True,
        progress=True,
        device=device)

    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, data_splits["train"], optimizer, criterion, epoch)
        test(model, device, data_splits["test"])
        scheduler.step()

    torch.save(model.state_dict(), "{}_checkpoint.pt".format(args.arch))


if __name__ == "__main__":
    main()
