
import torch

__all__ = ["mb3_vww_eval"]


def mb3_vww_eval(model, dataloader):
    size = len(dataloader.dataset)
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%\n")
    return {"acc": correct}
