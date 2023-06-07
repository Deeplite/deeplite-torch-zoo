from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY
from deeplite_torch_zoo.utils import load_pretrained_weights

NUM_IMAGENET_CLASSES = 1000


def make_wrapper_func(
    wrapper_fn,
    wrapper_name,
    model_name,
    dataset_name='imagenet',
    num_classes=NUM_IMAGENET_CLASSES,
    has_checkpoint=True,
):
    @MODEL_WRAPPER_REGISTRY.register(
        model_name=model_name,
        dataset_name=dataset_name,
        task_type='classification',
        has_checkpoint=has_checkpoint,
    )
    def wrapper_func(pretrained=False, num_classes=num_classes, device='cuda'):
        return wrapper_fn(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            device=device,
        )

    wrapper_func.__name__ = wrapper_name
    return wrapper_func


def load_checkpoint(model, model_name, dataset_name, model_urls, device):
    if f"{model_name}_{dataset_name}" not in model_urls:
        raise ValueError(
            f'Could not find a pretrained checkpoint for model {model_name} on dataset {dataset_name}. \n'
            'Use pretrained=False if you want to create a untrained model.'
        )
    checkpoint_url = model_urls[f"{model_name}_{dataset_name}"]
    model = load_pretrained_weights(model, checkpoint_url, device)
    return model
