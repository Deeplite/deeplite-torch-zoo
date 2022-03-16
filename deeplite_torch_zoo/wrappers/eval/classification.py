
import torch
from deeplite_torch_zoo.wrappers.registries import EVAL_WRAPPER_REGISTRY
__all__ = ["classification_eval"]

@EVAL_WRAPPER_REGISTRY.register('classification')
def classification_eval(model, dataloader, device="cuda"):
    size = len(dataloader.dataset)
    if device == "cuda":
        model = model.cuda()
    model.eval()
    running_corrects = 0
    with torch.set_grad_enabled(False):
        for X, y in dataloader:
            if device == "cuda":
                X = X.cuda()
                y = y.cuda()
            else:
                X = X.cpu()
                y = y.cpu()
            preds = model(X.cuda())
            _, preds = torch.max(preds, 1)
            running_corrects += torch.sum(preds == y.data)
    correct = running_corrects.double() / size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%\n")
    return {"acc": correct}
