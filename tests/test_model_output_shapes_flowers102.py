import pytest
import torch

from deeplite_torch_zoo import (create_model, get_model_by_name,
                                get_models_by_dataset)

TEST_BATCH_SIZE = 2
TEST_NUM_CLASSES = 42

FLOWERS_MODEL_TESTS = []
for model_name in get_models_by_dataset('flowers102'):
    FLOWERS_MODEL_TESTS.append((model_name, 'flowers102', 224, 3, 102))


@pytest.mark.slow
@pytest.mark.parametrize(
    ('model_name', 'dataset_name', 'input_resolution', 'num_inp_channels', 'target_output_shape'),
    FLOWERS_MODEL_TESTS,
)
def test_classification_model_output_shape(model_name, dataset_name, input_resolution,
    num_inp_channels, target_output_shape, download_checkpoint=False):
    model = get_model_by_name(
        model_name=model_name,
        dataset_name=dataset_name,
        pretrained=download_checkpoint,
        device='cpu',
    )
    model.eval()
    y = model(torch.randn(TEST_BATCH_SIZE, num_inp_channels, input_resolution, input_resolution))
    y.sum().backward()
    assert y.shape == (TEST_BATCH_SIZE, target_output_shape)


@pytest.mark.slow
@pytest.mark.parametrize(
    ('model_name', 'dataset_name', 'input_resolution', 'num_inp_channels', 'target_output_shape'),
    FLOWERS_MODEL_TESTS,
)
def test_classification_model_output_shape_arbitrary_num_clases(model_name, dataset_name, input_resolution,
    num_inp_channels, target_output_shape, download_checkpoint=False):
    model = create_model(
        model_name=model_name,
        num_classes=TEST_NUM_CLASSES,
        pretraining_dataset=dataset_name,
        pretrained=download_checkpoint,
        device='cpu',
    )
    model.eval()
    y = model(torch.randn(TEST_BATCH_SIZE, num_inp_channels, input_resolution, input_resolution))
    y.sum().backward()
    assert y.shape == (TEST_BATCH_SIZE, TEST_NUM_CLASSES)
