import torch
from tqdm import tqdm

from deeplite_torch_zoo.utils import switch_train_mode
from deeplite_torch_zoo.wrappers.registries import EVAL_WRAPPER_REGISTRY

__all__ = ['classification_eval']


@EVAL_WRAPPER_REGISTRY.register(task_type='classification')
def classification_eval(model, dataloader, progressbar=False, device='cuda', top_k=5, break_iter=None):
    if not torch.cuda.is_available():
        device = 'cpu'

    model.to(device)
    pred = []
    targets = []
    with switch_train_mode(model, is_training=False):
        with torch.no_grad():
            for iter_no, (inputs, target) in tqdm(enumerate(dataloader), disable=not progressbar):
                if iter_no == break_iter:
                    break
                inputs = inputs.to(device)
                target = target.to(device)
                y = model(inputs)

                pred.append(y.argsort(1, descending=True)[:, :top_k])
                targets.append(target)

    pred, targets = torch.cat(pred), torch.cat(targets)
    correct = (targets[:, None] == pred).float()
    acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)  # (top1, top5) accuracy
    top1, top5 = acc.mean(0).tolist()

    return {'acc': top1, 'acc_top5': top5}
