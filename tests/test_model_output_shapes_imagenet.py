import pytest
import torch

from deeplite_torch_zoo import get_model, list_models_by_dataset
from deeplite_torch_zoo.api.models.classification.model_implementation_dict import (
    FIXED_SIZE_INPUT_MODELS, INPLACE_ABN_MODELS)

TEST_BATCH_SIZE = 2
TEST_NUM_CLASSES = 42

IMAGENET_MODEL_TESTS = []
for model_name in list_models_by_dataset('imagenet'):
    if model_name not in FIXED_SIZE_INPUT_MODELS and model_name not in INPLACE_ABN_MODELS:
        IMAGENET_MODEL_TESTS.append((model_name, 'imagenet', 224, 3, 1000))


@pytest.mark.slow
@pytest.mark.parametrize(
    ('model_name', 'dataset_name', 'input_resolution', 'num_inp_channels', 'target_output_shape'),
    IMAGENET_MODEL_TESTS,
)
def test_classification_model_output_shape(model_name, dataset_name, input_resolution,
    num_inp_channels, target_output_shape, download_checkpoint=False):
    model = get_model(
        model_name=model_name,
        dataset_name=dataset_name,
        pretrained=download_checkpoint,
    )
    model.eval()
    y = model(torch.randn(TEST_BATCH_SIZE, num_inp_channels, input_resolution, input_resolution))
    y.sum().backward()
    assert y.shape == (TEST_BATCH_SIZE, target_output_shape)


@pytest.mark.slow
@pytest.mark.parametrize(
    ('model_name', 'dataset_name', 'input_resolution', 'num_inp_channels', 'target_output_shape'),
    IMAGENET_MODEL_TESTS,
)
def test_classification_model_output_shape_arbitrary_num_clases(model_name, dataset_name, input_resolution,
    num_inp_channels, target_output_shape, download_checkpoint=False):
    model = get_model(
        model_name=model_name,
        num_classes=TEST_NUM_CLASSES,
        dataset_name=dataset_name,
        pretrained=download_checkpoint,
    )
    model.eval()
    y = model(torch.randn(TEST_BATCH_SIZE, num_inp_channels, input_resolution, input_resolution))
    y.sum().backward()
    assert y.shape == (TEST_BATCH_SIZE, TEST_NUM_CLASSES)
