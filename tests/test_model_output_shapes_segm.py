from pathlib import Path

import pytest
import torch
from deeplite_torch_zoo import (create_model, get_data_splits_by_name,
                                get_model_by_name)

MOCK_DATASETS_PATH = Path('tests/fixture/datasets')
MOCK_VOC_PATH = MOCK_DATASETS_PATH / 'VOCdevkit'
MOCK_CARVANA_PATH = MOCK_DATASETS_PATH / 'carvana'

TEST_NUM_CLASSES = 42


@pytest.mark.parametrize(
    ('model_name', 'dataset_name', 'datasplit_kwargs', 'output_shape'),
    [
        ('deeplab_mobilenet', 'voc', {'backbone': 'vgg'}, (1, 21)),
        ('fcn32', 'voc', {'backbone': 'vgg'}, (1, 21)),
        ('unet_scse_resnet18', 'carvana', {}, (1, )),
        ('unet_scse_resnet18', 'voc', {}, (1, 21)),
        ('unet', 'carvana', {}, (1, )),
    ],
)
def test_segmentation_model_output_shape(model_name, dataset_name, datasplit_kwargs, output_shape):
    model = get_model_by_name(
        model_name=model_name,
        dataset_name=dataset_name,
        pretrained=True,
        progress=False,
        device="cpu",
    )
    if 'unet' in model_name:
        model_name = 'unet'
    test_loader = get_data_splits_by_name(
        data_root=MOCK_DATASETS_PATH if 'voc' in dataset_name else MOCK_CARVANA_PATH,
        dataset_name=dataset_name,
        model_name=model_name,
        num_workers=0,
        device="cpu",
        **datasplit_kwargs,
    )["test"]
    dataset = test_loader.dataset
    if 'unet' in model_name:
        img, msk, _ = dataset[0]
    else:
        img, msk = dataset[0]
    model.eval()
    y = model(torch.unsqueeze(img, dim=0))
    assert y.shape == (*output_shape, *msk.shape)


@pytest.mark.parametrize(
    ('model_name', 'dataset_name', 'datasplit_kwargs', 'output_shape'),
    [
        ('deeplab_mobilenet', 'voc', {'backbone': 'vgg'}, (1, TEST_NUM_CLASSES)),
        ('fcn32', 'voc', {'backbone': 'vgg'}, (1, TEST_NUM_CLASSES)),
    ],
)
def test_create_segmentation_model_output_shape(model_name, dataset_name, datasplit_kwargs, output_shape):
    model = create_model(
        model_name=model_name,
        pretraining_dataset=dataset_name,
        num_classes=TEST_NUM_CLASSES,
        progress=False,
        device="cpu",
    )
    if 'unet' in model_name:
        model_name = 'unet'
    test_loader = get_data_splits_by_name(
        data_root=MOCK_DATASETS_PATH if 'voc' in dataset_name else MOCK_CARVANA_PATH,
        dataset_name=dataset_name,
        model_name=model_name,
        num_workers=0,
        device="cpu",
        **datasplit_kwargs,
    )["test"]
    dataset = test_loader.dataset
    if 'unet' in model_name:
        img, msk, _ = dataset[0]
    else:
        img, msk = dataset[0]
    model.eval()
    y = model(torch.unsqueeze(img, dim=0))
    assert y.shape == (*output_shape, *msk.shape)
