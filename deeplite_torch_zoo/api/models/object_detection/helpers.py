from pathlib import Path
import urllib.parse as urlparse

import deeplite_torch_zoo
from deeplite_torch_zoo.utils import load_pretrained_weights
from deeplite_torch_zoo.api.models.object_detection.checkpoints import (
    CHECKPOINT_STORAGE_URL,
    model_urls,
)
from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY


DATASET_LIST = [
    ('person_detection', 1),
    ('voc', 20),
    ('coco', 80),
    ('voc07', 20),
    ('custom_person_detection', 1),
]


def get_project_root() -> Path:
    return Path(deeplite_torch_zoo.__file__).parents[1]


def load_pretrained_model(model, model_name, dataset_name, device='cpu'):
    if f"{model_name}_{dataset_name}" not in model_urls:
        raise ValueError(
            f'Could not find a pretrained checkpoint for model {model_name} on dataset {dataset_name}. \n'
            'Use pretrained=False if you want to create a untrained model.'
        )
    checkpoint_url = urlparse.urljoin(
        CHECKPOINT_STORAGE_URL, model_urls[f'{model_name}_{dataset_name}']
    )
    model = load_pretrained_weights(model, checkpoint_url, device)
    return model


def make_wrapper_func(
    wrapper_generator_fn,
    wrapper_name,
    model_name,
    dataset_name,
    num_classes,
    **default_kwargs,
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
        **kwargs,
    ):
        default_kwargs.update(kwargs)
        return wrapper_generator_fn(
            model_name=model_name,
            dataset_name=dataset_name,
            num_classes=num_classes,
            pretrained=pretrained,
            **default_kwargs,
        )

    wrapper_func.__name__ = wrapper_name
    return wrapper_func
