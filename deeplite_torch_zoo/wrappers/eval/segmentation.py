from deeplite_torch_zoo.src.segmentation.deeplab.eval import evaluate_deeplab
from deeplite_torch_zoo.src.segmentation.fcn.eval import evaluate_fcn
from deeplite_torch_zoo.src.segmentation.Unet.eval import eval_net, eval_net_miou

__all__ = ["seg_eval_func"]

def seg_eval_func(model, data_loader, net="unet", device="cuda"):
    if net == "unet":
        dice_coeff = eval_net(model, loader=data_loader, device=device)
        return {"dice_coeff": dice_coeff}

    if "unet_scse" in net:
        miou = eval_net_miou(model, loader=data_loader, device=device)
        return {"miou": miou}

    if "fcn" in net:
        return {"miou": evaluate_fcn(model, loader=data_loader, device=device)}

    if "deeplab" in net:
        return {"miou": evaluate_deeplab(model, loader=data_loader, device=device)}

    raise ValueError
