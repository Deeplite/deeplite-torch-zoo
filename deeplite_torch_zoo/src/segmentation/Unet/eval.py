import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from deeplite_torch_zoo.src.segmentation.eval.utils.metrics import \
    compute_iou_batch
from deeplite_torch_zoo.src.segmentation.Unet.dice_loss import dice_coeff


def eval_net(net, loader, device="cuda"):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    for batch in loader:
        imgs, true_masks = batch[0], batch[1]
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=mask_type)

        with torch.no_grad():
            mask_pred = net(imgs)

        if net.n_classes > 1:
            tot += F.cross_entropy(mask_pred, true_masks).item()
        else:
            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.5).float()
            tot += dice_coeff(pred, true_masks).item()

    net.train()
    return tot / n_val


def eval_net_miou(model, loader, device="cuda", net_type="unet"):
    """Evaluation without the densecrf with the dice coefficient"""
    num_classes = loader.dataset.num_classes
    assert num_classes > 1
    classes = np.arange(1, num_classes)
    model.eval()
    model.to(device)
    tot = 0
    test_ious = []

    for batch in loader:
        imgs, labels = batch[0], batch[1]
        imgs = imgs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(imgs)

        labels_np = labels.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()

        if num_classes == 2:
            preds_np = torch.sigmoid(preds).detach().cpu().numpy() > 0.5
            preds_np = np.squeeze(preds_np, axis=1)
            if len(labels_np.shape) == 4:
                labels_np = np.squeeze(labels_np, axis=1)
        else:
            preds_np = np.argmax(preds_np, axis=1)
        iou = compute_iou_batch(preds_np, labels_np, classes)

        test_ious.append(iou)

    test_iou = np.nanmean(test_ious)
    model.train()
    return test_iou
