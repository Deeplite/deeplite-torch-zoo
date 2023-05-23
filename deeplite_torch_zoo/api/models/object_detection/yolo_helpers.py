import urllib.parse as urlparse

from deeplite_torch_zoo.src.object_detection.yolov5.yolov5 import YOLOModel
from deeplite_torch_zoo.utils import load_pretrained_weights
from deeplite_torch_zoo.api.models.object_detection.yolo_checkpoints import (
    CHECKPOINT_STORAGE_URL,
    model_urls,
)
from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY


def create_yolo_model(
    model_name="yolo5s",
    dataset_name="voc",
    num_classes=20,
    config_path=None,
    pretrained=False,
    progress=True,
    device="cuda",
    **kwargs,
):  # pylint: disable=W0621
    model = YOLOModel(
        config_path,
        nc=num_classes,
        **kwargs,
    )
    if pretrained:
        if f"{model_name}_{dataset_name}" not in model_urls:
            raise ValueError(
                f'Could not find a pretrained checkpoint for model {model_name} on dataset {dataset_name}. \n'
                'Use pretrained=False if you want to create a untrained model.'
            )
        checkpoint_url = urlparse.urljoin(
            CHECKPOINT_STORAGE_URL, model_urls[f"{model_name}_{dataset_name}"]
        )
        model = load_pretrained_weights(model, checkpoint_url, progress, device)
    return model.to(device)


def make_wrapper_func(
    wrapper_name, model_name, dataset_name, num_classes, config_path, **default_kwargs
):  # pylint: disable=W0621
    has_checkpoint = True
    if f"{model_name}_{dataset_name}" not in model_urls:
        has_checkpoint = False

    @MODEL_WRAPPER_REGISTRY.register(
        model_name=model_name,
        dataset_name=dataset_name,
        task_type='object_detection',
        has_checkpoint=has_checkpoint,
    )
    def wrapper_func(
        pretrained=False,
        num_classes=num_classes,
        progress=True,
        device="cuda",
        **kwargs,
    ):
        default_kwargs.update(kwargs)
        return create_yolo_model(
            model_name=model_name,
            dataset_name=dataset_name,
            num_classes=num_classes,
            pretrained=pretrained,
            config_path=config_path,
            progress=progress,
            device=device,
            **default_kwargs,
        )

    wrapper_func.__name__ = wrapper_name
    return wrapper_func
