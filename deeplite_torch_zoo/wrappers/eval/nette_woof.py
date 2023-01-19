import torch
from tqdm import tqdm

from deeplite_torch_zoo.utils import training_mode_switcher
from deeplite_torch_zoo.wrappers.registries import EVAL_WRAPPER_REGISTRY

__all__ = ['classification_eval']

imagenette_map = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
imagewoof_map =  [155, 159, 162, 167, 182, 193, 207, 229, 258, 273]
@EVAL_WRAPPER_REGISTRY.register(task_type='classification')
def classification_eval(model, dataloader, progressbar=False, device='cuda', top_k=5):
    if not torch.cuda.is_available():
        device = 'cpu'

    model.to(device)
    pred = []
    targets = []
    with training_mode_switcher(model, is_training=False):
        with torch.no_grad():
            for inputs, target in tqdm(dataloader, disable=not progressbar):
                inputs = inputs.to(device)
                target = torch.tensor([imagenette_map[label] for label in target])
                target = target.to(device)
                y = model(inputs)
                pred.append(y.argsort(1, descending=True)[:, :top_k])
                targets.append(target)

    pred, targets = torch.cat(pred), torch.cat(targets)
    correct = (targets[:, None] == pred).float()
    acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)  # (top1, top5) accuracy
    top1, top5 = acc.mean(0).tolist()

    return {'acc': top1, 'acc_top5': top5}
