import torch

from deeplite_torch_zoo.src.segmentation.fcn.utils import label_accuracy_score


def evaluate_fcn(model, loader, device="cuda"):
    model.eval()

    n_class = len(loader.dataset.class_names)
    label_trues, label_preds = [], []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            score = model(data)

            imgs = data.data.cpu()
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu()
            for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                img, lt = loader.dataset.untransform(img, lt)
                label_trues.append(lt)
                label_preds.append(lp)
    metrics = label_accuracy_score(label_trues, label_preds, n_class)
    return metrics[1]
