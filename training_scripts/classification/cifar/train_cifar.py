# Source: https://github.com/y0ast/pytorch-snippets/blob/main/minimal_cifar/train_cifar.py

import argparse

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from deeplite_torch_zoo import get_dataloaders, get_model
from deeplite_torch_zoo.utils import LOGGER


def train_epoch(model, train_loader, optimizer, epoch, writer, dryrun=False):
    model.train()

    total_loss = []

    for data, target in tqdm(train_loader):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        optimizer.zero_grad()

        prediction = model(data)
        loss = F.nll_loss(F.log_softmax(prediction, dim=1), target)

        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

        if dryrun:
            break

    avg_loss = sum(total_loss) / len(total_loss)

    writer.add_scalar('train/avg_loss', avg_loss, epoch)

    LOGGER.info(f"Epoch: {epoch}:")
    LOGGER.info(f"Train Set: Average Loss: {avg_loss:.2f}")


def test(model, test_loader, epoch, writer):
    model.eval()

    loss = 0
    correct = 0

    for data, target in test_loader:
        with torch.no_grad():
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            prediction = model(data)
            loss += F.nll_loss(F.log_softmax(prediction, dim=1), target, reduction="sum")

            prediction = prediction.max(1)[1]
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    loss /= len(test_loader.dataset)

    percentage_correct = 100.0 * correct / len(test_loader.dataset)

    LOGGER.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
            loss, correct, len(test_loader.dataset), percentage_correct
        )
    )

    writer.add_scalar('test/avg_loss', loss, epoch)
    writer.add_scalar('test/accuracy', percentage_correct, epoch)

    return loss, percentage_correct


def train(args, model=None, data_splits=None):
    torch.manual_seed(args.seed)

    writer = SummaryWriter(comment=f'{args.model}')

    data_splits = get_dataloaders(
        data_root=args.data_root,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        num_workers=args.workers
    )
    train_loader, test_loader = data_splits['train'], data_splits['test']

    model = get_model(
        model_name=args.model,
        dataset_name=args.dataset_name,
        pretrained=False,
    )

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.lr_gamma
    )

    for epoch in range(1, args.epochs + 1):
        train_epoch(model, train_loader, optimizer, epoch, writer, dryrun=args.dryrun)
        if args.dryrun:
            break
        test(model, test_loader, epoch, writer, dryrun=args.dryrun)
        scheduler.step()


    if not args.dryrun:
        torch.save(model.state_dict(), f"{args.model}_cifar100.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate (default: 0.1)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--milestones', nargs='+', type=int, default=[60, 120, 160],
                        help='Milestone epochs for LR schedule (default: [60, 120, 160])')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum value (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (default: 5e-4)')
    parser.add_argument('--lr_gamma', type=float, default=0.2, help='LR gamma (default: 0.2)')
    parser.add_argument('--workers', type=int, default=8, help='Number of data loading workers (default: 8)')
    parser.add_argument('--model', type=str, default='resnet50', help='Model architecture (default: resnet50)')
    parser.add_argument('--dataset_name', type=str, default='cifar100', help='Name of the dataset (default: cifar100)')
    parser.add_argument('--data_root', type=str, default='./', help='Root directory of the dataset (default: ./)')

    parser.add_argument('--dryrun', action='store_true', help='Dry run mode for testing')

    args = parser.parse_args()
    train(args)
