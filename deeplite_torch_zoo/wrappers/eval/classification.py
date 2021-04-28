
import torch

__all__ = ["classification_eval"]


def classification_eval(model, dataloader):
    size = len(dataloader.dataset)
    model.eval()
    running_corrects = 0
    with torch.set_grad_enabled(False):
        for X, y in dataloader:
            preds = model(X)
            _, preds = torch.max(preds, 1)
            running_corrects += torch.sum(preds == y.data)
    correct = running_corrects.double() / size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%\n")
    return {"acc": correct}
