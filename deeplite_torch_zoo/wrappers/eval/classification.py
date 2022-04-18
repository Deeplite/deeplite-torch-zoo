import torch
from torchmetrics import Accuracy
from deeplite_torch_zoo.wrappers.registries import EVAL_WRAPPER_REGISTRY


__all__ = ["classification_eval"]


@EVAL_WRAPPER_REGISTRY.register(task_type='classification')
def classification_eval(model, dataloader, device="cuda"):
    metrics = {
        'acc': Accuracy(),
        'acc_top5': Accuracy(top_k=5),
    }

    if device == "cuda":
        model = model.cuda()
        for metric_name in metrics:
            metrics[metric_name] = metrics[metric_name].cuda()

    model.eval()
    with torch.set_grad_enabled(False):
        for X, y in dataloader:
            if device == "cuda":
                X = X.cuda()
                y = y.cuda()
            else:
                X = X.cpu()
                y = y.cpu()

            pred = model(X)
            for metric_name, metric_fn in metrics.items():
                metric_fn.update(pred, y.data)

    for metric_name in metrics:
        metrics[metric_name] = metrics[metric_name].compute()
    return metrics
