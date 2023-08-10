# Source: https://github.com/y0ast/pytorch-snippets/blob/main/minimal_cifar/train_cifar.py

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from deeplite_torch_zoo import get_dataloaders, get_model
from deeplite_torch_zoo.utils import LOGGER


def train(model, train_loader, optimizer, epoch, writer):
    model.train()

    total_loss = []

    for data, target in tqdm(train_loader):
        data = data.cuda()
        target = target.cuda()

        optimizer.zero_grad()

        prediction = model(data)
        loss = F.nll_loss(F.log_softmax(prediction, dim=1), target)

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

    avg_loss = sum(total_loss) / len(total_loss)

    writer.add_scalar('train/avg_loss', avg_loss, epoch)

    LOGGER.info((f"Epoch: {epoch}:")
    LOGGER.info((f"Train Set: Average Loss: {avg_loss:.2f}")


def test(model, test_loader, epoch, writer):
    model.eval()

    loss = 0
    correct = 0

    for data, target in test_loader:
        with torch.no_grad():
            data = data.cuda()
            target = target.cuda()

            prediction = model(data)
            loss += F.nll_loss(F.log_softmax(prediction, dim=1), target, reduction="sum")

            prediction = prediction.max(1)[1]
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    loss /= len(test_loader.dataset)

    percentage_correct = 100.0 * correct / len(test_loader.dataset)

    LOGGER.info((
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
            loss, correct, len(test_loader.dataset), percentage_correct
        )
    )

    writer.add_scalar('test/avg_loss', loss, epoch)
    writer.add_scalar('test/accuracy', percentage_correct, epoch)

    return loss, percentage_correct


class CIFARConfig:
    epochs = 200
    lr = 0.1
    seed = 1
    batch_size = 128
    milestones = [60, 120, 160]
    momentum = 0.9
    weight_decay = 5e-4
    lr_gamma = 0.2
    workers = 8
    model = 'resnet50'
    dataset_name = 'cifar100'
    data_root = './'


def train(args: CIFARConfig, model=None, data_splits=None):
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
        pretrained=False
    )

    model = model.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.lr_gamma
    )

    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, epoch, writer)
        test(model, test_loader, epoch, writer)
        scheduler.step()

    torch.save(model.state_dict(), f"{args.model}_cifar100.pt")


if __name__ == "__main__":
    train(CIFARConfig())
