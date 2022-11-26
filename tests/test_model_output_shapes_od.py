from pathlib import Path

import pytest
import torch
from deeplite_torch_zoo import (create_model, get_data_splits_by_name,
                                get_model_by_name, list_models)
from deeplite_torch_zoo.wrappers.registries import MODEL_WRAPPER_REGISTRY


def get_models_by_dataset(dataset_name):
    return [model_key.model_name for model_key in
        list_models(dataset_name, return_list=True, print_table=False, include_no_checkpoint=True)
        if model_key.dataset_name == dataset_name]


MOCK_DATASETS_PATH = Path('tests/fixture/datasets')
MOCK_VOC_PATH = MOCK_DATASETS_PATH / 'VOCdevkit'
MOCK_PERSONDET_PATH = MOCK_DATASETS_PATH / 'person_detection'

TEST_BATCH_SIZE = 2
TEST_NUM_CLASSES = 42

BLACKLIST = ('yolo5_6sa', 'yolo3_tiny', 'yolo_fdresnet18x0.25', 'yolo_fdresnet18x0.5')

DETECTION_MODEL_TESTS = [
    ('mb1_ssd', 'voc', {'num_classes': 20}, [(3000, ), (3000, 4)], MOCK_VOC_PATH, True),
    ('mb2_ssd_lite', 'voc', {'num_classes': 20}, [(3000, ), (3000, 4)], MOCK_VOC_PATH, True),
    ('mb2_ssd', 'voc', {'num_classes': 20}, [(3000, ), (3000, 4)], MOCK_VOC_PATH, True),
    ('resnet18_ssd', 'voc', {'num_classes': 20}, [(8732, ), (8732, 4)], MOCK_VOC_PATH, True),
    ('resnet34_ssd', 'voc', {'num_classes': 20}, [(8732, ), (8732, 4)], MOCK_VOC_PATH, True),
    ('resnet50_ssd', 'voc', {'num_classes': 20}, [(8732, ), (8732, 4)], MOCK_VOC_PATH, True),
]


for model_name in get_models_by_dataset('voc'):
    if 'yolo' in model_name and model_name not in BLACKLIST:
        download_checkpoint = False
        if model_name in MODEL_WRAPPER_REGISTRY.pretrained_models:
            download_checkpoint = True
        DETECTION_MODEL_TESTS.append((model_name, 'voc', {'img_size': 448, 'num_classes': 20},
            [(3, 56, 56), (3, 28, 28), (3, 14, 14)], MOCK_VOC_PATH, download_checkpoint))


for model_name in get_models_by_dataset('person_detection'):
    if 'yolo' in model_name and model_name not in BLACKLIST:
        download_checkpoint = False
        if model_name in MODEL_WRAPPER_REGISTRY.pretrained_models:
            download_checkpoint = True
        DETECTION_MODEL_TESTS.append((model_name, 'person_detection', {'img_size': 320, 'num_classes': 1},
            [(3, 40, 40), (3, 20, 20), (3, 10, 10)], MOCK_PERSONDET_PATH, download_checkpoint))


@pytest.mark.parametrize(
    ('model_name', 'dataset_name', 'datasplit_kwargs', 'output_shapes',
     'mock_dataset_path', 'download_checkpoint'),
    DETECTION_MODEL_TESTS
)
def test_detection_model_output_shape(model_name, dataset_name, datasplit_kwargs,
    output_shapes, mock_dataset_path, download_checkpoint):
    model = get_model_by_name(
        model_name=model_name,
        dataset_name=dataset_name,
        pretrained=download_checkpoint,
        device='cpu',
    )
    train_loader = get_data_splits_by_name(
        data_root=mock_dataset_path,
        dataset_name=dataset_name,
        model_name=model_name,
        batch_size=TEST_BATCH_SIZE,
        num_workers=0,
        device='cpu',
        **datasplit_kwargs,
    )['train']

    if 'yolo' in model_name:
        dataset = train_loader.dataset
        img, _, _, _ = dataset[0]
        y = model(torch.unsqueeze(img, dim=0))
        y[0].sum().backward()
        assert y[0].shape == (1, *output_shapes[0], datasplit_kwargs['num_classes'] + 5)
        assert y[1].shape == (1, *output_shapes[1], datasplit_kwargs['num_classes'] + 5)
        assert y[2].shape == (1, *output_shapes[2], datasplit_kwargs['num_classes'] + 5)
    else:
        img, _, _ = next(iter(train_loader))
        model.eval()
        y1, y2 = model(img)
        y1.sum().backward()
        assert y1.shape == (TEST_BATCH_SIZE, *output_shapes[0], datasplit_kwargs['num_classes'] + 1)
        assert y2.shape == (TEST_BATCH_SIZE, *output_shapes[1])


@pytest.mark.parametrize(
    ('model_name', 'dataset_name', 'datasplit_kwargs', 'output_shapes',
     'mock_dataset_path', 'download_checkpoint'),
    DETECTION_MODEL_TESTS
)
def test_detection_model_output_shape_arbitrary_num_clases(model_name, dataset_name, datasplit_kwargs,
    output_shapes, mock_dataset_path, download_checkpoint):
    model = create_model(
        model_name=model_name,
        num_classes=TEST_NUM_CLASSES,
        pretraining_dataset=dataset_name,
        pretrained=download_checkpoint,
        device='cpu',
    )
    train_loader = get_data_splits_by_name(
        data_root=mock_dataset_path,
        dataset_name=dataset_name,
        model_name=model_name,
        batch_size=TEST_BATCH_SIZE,
        num_workers=0,
        device='cpu',
        **datasplit_kwargs,
    )['train']

    if 'yolo' in model_name:
        dataset = train_loader.dataset
        img, _, _, _ = dataset[0]
        y = model(torch.unsqueeze(img, dim=0))
        y[0].sum().backward()
        assert y[0].shape == (1, *output_shapes[0], TEST_NUM_CLASSES + 5)
        assert y[1].shape == (1, *output_shapes[1], TEST_NUM_CLASSES + 5)
        assert y[2].shape == (1, *output_shapes[2], TEST_NUM_CLASSES + 5)
    else:
        img, _, _ = next(iter(train_loader))
        model.eval()
        y1, y2 = model(img)
        y1.sum().backward()
        assert y1.shape == (TEST_BATCH_SIZE, *output_shapes[0], TEST_NUM_CLASSES + 1)
        assert y2.shape == (TEST_BATCH_SIZE, *output_shapes[1])
