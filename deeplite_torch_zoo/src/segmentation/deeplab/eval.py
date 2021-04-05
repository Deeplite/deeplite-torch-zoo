import numpy as np
import torch

from deeplite_torch_zoo.src.segmentation.deeplab.utils.metrics import Evaluator


def evaluate_deeplab(model, loader, device="cuda"):
    model.eval()
    if "cuda" in device:
        model.cuda()
    evaluator = Evaluator(loader.dataset.num_classes)
    for i, (image, target) in enumerate(loader):
        image, target = image.to(device), target.to(device)
        with torch.no_grad():
            output = model(image)
        pred = output.data.cpu().numpy()
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        # Add batch sample into evaluator
        evaluator.add_batch(target, pred)

    # Fast test during the training
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    return mIoU
