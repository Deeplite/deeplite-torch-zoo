from deeplite_torch_zoo.src.segmentation.deeplab.eval import evaluate_deeplab
from deeplite_torch_zoo.src.segmentation.fcn.eval import evaluate_fcn
from deeplite_torch_zoo.src.segmentation.Unet.eval import eval_net, eval_net_miou

from deeplite_torch_zoo.wrappers.registries import EVAL_WRAPPER_REGISTRY

__all__ = ["unet_eval_func", "unet_scse_eval_func","fcn_eval_func","deeplab_eval_func"]

@EVAL_WRAPPER_REGISTRY.register('segmentation','unet', 'general_seg_dataset')
def unet_eval_func(model, data_loader, device="cuda"):
    dice_coeff = eval_net(model, loader=data_loader, device=device)
    return {"dice_coeff": dice_coeff}

@EVAL_WRAPPER_REGISTRY.register('segmentation','unet_scse', 'general_seg_dataset')
def unet_scse_eval_func(model, data_loader, device="cuda"):
    miou = eval_net_miou(model, loader=data_loader, device=device)
    return {"miou": miou}

@EVAL_WRAPPER_REGISTRY.register('segmentation','fcn', 'general_seg_dataset')
def fcn_eval_func(model, data_loader, device="cuda"):
    return {"miou": evaluate_fcn(model, loader=data_loader, device=device)}

@EVAL_WRAPPER_REGISTRY.register('segmentation','deeplab', 'general_seg_dataset')
def deeplab_eval_func(model, data_loader, device="cuda"):
    miou = {"miou": evaluate_deeplab(model, loader=data_loader, device=device)}
    return miou