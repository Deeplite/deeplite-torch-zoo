from collections import namedtuple

import pytest
import torch

from deeplite_torch_zoo import (create_model, get_model,
                                list_models_by_dataset)

Dataset = namedtuple(typename='Dataset', field_names=('name', 'img_res', 'in_channels', 'num_classes'))

TEST_BATCH_SIZE = 2
TEST_NUM_CLASSES = 42

OVERRIDE_TEST_PARAMS = {
    'vww': {'mobilenet_v1_0.25_96px': {'img_res': 96}}
}

DATASETS = [
    Dataset('tinyimagenet', 64, 3, 100),
    Dataset('imagenet16', 224, 3, 16),
    Dataset('imagenet10', 224, 3, 10),
    Dataset('vww', 224, 3, 2),
    Dataset('cifar10', 32, 3, 10),
    Dataset('cifar100', 32, 3, 100),
    Dataset('mnist', 28, 1, 10),
]

CLASSIFICATION_MODEL_TESTS = []
for dataset in DATASETS:
    for model_name in list_models_by_dataset(dataset.name):
        test_params = {
            'model_name': model_name,
            'dataset_name': dataset.name,
            'img_res': dataset.img_res,
            'in_channels': dataset.in_channels,
            'num_classes': dataset.num_classes,
        }
        if dataset.name in OVERRIDE_TEST_PARAMS and model_name in OVERRIDE_TEST_PARAMS[dataset.name]:
            test_params.update(OVERRIDE_TEST_PARAMS[dataset.name][model_name])
        CLASSIFICATION_MODEL_TESTS.append(tuple(test_params.values()))


IMAGENET_MODEL_NAMES = [
    # torchvision:
    'mobilenet_v3_small',
    'squeezenet1_0',
    # timm:
    'hrnet_w18_small_v2',
    'efficientnet_es_pruned',
    # pytorchcv:
    'fdmobilenet_wd4',
    'proxylessnas_mobile',
    # zoo:
    'mobilenetv2_w035',
    'mobileone_s0',
    'mobileone_s4',
]

for model_name in IMAGENET_MODEL_NAMES:
    CLASSIFICATION_MODEL_TESTS.append((model_name, 'imagenet', 224, 3, 1000))


@pytest.mark.parametrize(
    ('model_name', 'dataset_name', 'input_resolution', 'num_inp_channels', 'target_output_shape'),
    CLASSIFICATION_MODEL_TESTS,
)
def test_classification_model_output_shape(model_name, dataset_name, input_resolution,
    num_inp_channels, target_output_shape, download_checkpoint=True):
    model = get_model(
        model_name=model_name,
        dataset_name=dataset_name,
        pretrained=download_checkpoint,
    )
    model.eval()
    y = model(torch.randn(TEST_BATCH_SIZE, num_inp_channels, input_resolution, input_resolution))
    y.sum().backward()
    assert y.shape == (TEST_BATCH_SIZE, target_output_shape)


@pytest.mark.parametrize(
    ('model_name', 'dataset_name', 'input_resolution', 'num_inp_channels', 'target_output_shape'),
    CLASSIFICATION_MODEL_TESTS,
)
def test_classification_model_output_shape_arbitrary_num_clases(model_name, dataset_name, input_resolution,
    num_inp_channels, target_output_shape, download_checkpoint=True):
    model = create_model(
        model_name=model_name,
        num_classes=TEST_NUM_CLASSES,
        pretraining_dataset=dataset_name,
        pretrained=download_checkpoint,
    )
    model.eval()
    y = model(torch.randn(TEST_BATCH_SIZE, num_inp_channels, input_resolution, input_resolution))
    y.sum().backward()
    assert y.shape == (TEST_BATCH_SIZE, TEST_NUM_CLASSES)
