from deeplite_torch_zoo.utils import load_pretrained_weights
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY
from deeplite_torch_zoo.src.classification.imagenet_models.mobileone import mobileone


__all__ = ["mobileone_s0", "mobileone_s1", "mobileone_s2", "mobileone_s3", "mobileone_s4"]


base_url = "https://docs-assets.developer.apple.com/ml-research/datasets/mobileone"
model_urls = {
    "s0": f"{base_url}/mobileone_s0_unfused.pth.tar",
    "s1": f"{base_url}/mobileone_s1_unfused.pth.tar",
    "s2": f"{base_url}/mobileone_s2_unfused.pth.tar",
    "s3": f"{base_url}/mobileone_s3_unfused.pth.tar",
    "s4": f"{base_url}/mobileone_s4_unfused.pth.tar",
}


def get_mobileone_variant(pretrained=False, progress=True, device="cuda", num_classes=1000, variant="s0"):

    model = mobileone(num_classes=num_classes, inference_mode = False, variant=variant )
    # model = reparameterize_model(model)
    if pretrained:
        print ("Loading pre-trained")
        checkpoint_url = model_urls[variant]
        print (checkpoint_url)
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)

@MODEL_WRAPPER_REGISTRY.register(model_name='mobileone_s0', dataset_name='imagenet', task_type='classification')
def mobileone_s0(pretrained=False, progress=True, device="cuda", num_classes=1000):
    return get_mobileone_variant(pretrained=pretrained, progress=progress, device=device, num_classes=num_classes, variant="s0")

@MODEL_WRAPPER_REGISTRY.register(model_name='mobileone_s1', dataset_name='imagenet', task_type='classification')
def mobileone_s1(pretrained=False, progress=True, device="cuda", num_classes=1000):
    return get_mobileone_variant(pretrained=pretrained, progress=progress, device=device, num_classes=num_classes, variant="s1")

@MODEL_WRAPPER_REGISTRY.register(model_name='mobileone_s2', dataset_name='imagenet', task_type='classification')
def mobileone_s2(pretrained=False, progress=True, device="cuda", num_classes=1000):
    return get_mobileone_variant(pretrained=pretrained, progress=progress, device=device, num_classes=num_classes, variant="s2")

@MODEL_WRAPPER_REGISTRY.register(model_name='mobileone_s3', dataset_name='imagenet', task_type='classification')
def mobileone_s3(pretrained=False, progress=True, device="cuda", num_classes=1000):
    return get_mobileone_variant(pretrained=pretrained, progress=progress, device=device, num_classes=num_classes, variant="s3")

@MODEL_WRAPPER_REGISTRY.register(model_name='mobileone_s4', dataset_name='imagenet', task_type='classification')
def mobileone_s4(pretrained=False, progress=True, device="cuda", num_classes=1000):
    return get_mobileone_variant(pretrained=pretrained, progress=progress, device=device, num_classes=num_classes, variant="s4")


if __name__ == "__main__":
    
    # Accuracy computed using deeplite-torch-zoo
    # mobileone_s0 {'acc': 0.713699996471405, 'acc_top5': 0.8986600041389465}
    # mobileone_s1 {'acc': 0.7576599717140198, 'acc_top5': 0.9276599884033203}
    # mobileone_s2 {'acc': 0.7739399671554565, 'acc_top5': 0.9362799525260925}
    # mobileone_s3 {'acc': 0.779259979724884, 'acc_top5': 0.9388999938964844}
    # mobileone_s4 {'acc': 0.792959988117218, 'acc_top5': 0.9438599944114685}
    
    from deeplite_torch_zoo import get_model_by_name, get_data_splits_by_name
    from deeplite_torch_zoo import get_eval_function
    data_splits = get_data_splits_by_name(
                data_root="/neutrino/datasets/imagenet",
                dataset_name="imagenet",
                model_name="mobileone_s0",
                batch_size=128,
                num_workers=1
            )
    train_loader, test_loader = data_splits['train'], data_splits['test']
    eval_fn = get_eval_function(
            model_name="mobileone_s0",
            dataset_name="imagenet",
        )
    for i in range(5):
        model_name = "mobileone_s"+str(i)
        model = get_model_by_name(model_name, dataset_name="imagenet", progress=True, pretrained=True)

        eval = eval_fn(model, test_loader, progressbar=False)
        print (model_name, eval)
