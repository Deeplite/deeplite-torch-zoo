import argparse
import os

import torch
from torch import nn
from deeplite_torch_zoo.src.classification.mobilenetv3 import mobilenetv3_large, mobilenetv3_small
from deeplite_torch_zoo.wrappers.wrapper import get_data_splits_by_name
from deeplite_torch_zoo.wrappers.models import mobilenetv3_small_vww, mobilenetv3_large_vww

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100*correct



def train(arch_type="small", BATCH_SIZE=128, learning_rate=1e-2):
    if arch_type == "small":
        model = mobilenetv3_small_vww(pretrained=True)
    elif arch_type == "large":
        model = mobilenetv3_large_vww(pretrained=True)
    else:
        raise ValueError

    datasplit = get_data_splits_by_name(
        dataset_name="vww",
        data_root="/neutrino/datasets/vww",
        batch_size=BATCH_SIZE,
        device="cuda",
    )
    train_dataloader = datasplit["train"]
    test_dataloader = datasplit["test"]
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epochs = 100
    for t in range(0, epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        ac = test_loop(test_dataloader, model, loss_fn)
        if t % 5 == 0:
            torch.save(model.state_dict(), f'mobilenetv3-vww/{arch_type}/mobilenetv3-{arch_type}-vww-{ac}.pth')
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="small", choices=["small", "large"])
    opt = parser.parse_args()
    os.makedirs(f"mobilenetv3-vww/{opt.arch}", exist_ok=True)
    train(opt.arch)








