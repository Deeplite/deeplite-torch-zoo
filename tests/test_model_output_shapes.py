from pathlib import Path

import pytest

import torch

from deeplite_torch_zoo import get_model_by_name, get_data_splits_by_name, create_model
from tests.mock_datasets import VocYoloFake


MOCK_DATASETS_PATH = Path('tests/fixture/datasets')
MOCK_VOC_PATH = MOCK_DATASETS_PATH / 'VOCdevkit'
MOCK_CARVANA_PATH = MOCK_DATASETS_PATH / 'carvana'

TEST_BATCH_SIZE = 2

MODEL_NAME_DATASPLIT_FN_ARG_MAP = {
    'mb2_ssd': 'mb2_ssd_lite',
    'yolo3': 'yolo',
    'yolo4s': 'yolo',
    'yolo4m': 'yolo',
    'yolo4l': 'yolo',
    'yolo4l_leaky': 'yolo',
    'yolo4x': 'yolo',
    'unet_scse_resnet18': 'unet',
}

DATASET_NAME_DATASPLIT_FN_ARG_MAP = {
    'voc': 'voc',
    'voc_1': 'voc',
    'voc_2': 'voc',
    'carvana': 'carvana',
    'person_detection_1': 'person_detection',
}

CLASSIFICATION_MODEL_TESTS = [
    ('resnet18', 'imagenet16', 224, 3, 16),
    ('resnet50', 'imagenet16', 224, 3, 16),
    ('resnet18', 'imagenet10', 224, 3, 10),
    ('mobilenet_v2_0_35', 'imagenet10', 224, 3, 10),
    ('resnet18', 'vww', 224, 3, 2),
    ('resnet50', 'vww', 224, 3, 2),
    ('mobilenet_v1', 'vww', 224, 3, 2),
    ('lenet5', 'mnist', 28, 1, 10),
    ('mlp2', 'mnist', 28, 1, 10),
    ('mlp4', 'mnist', 28, 1, 10),
    ('mlp8', 'mnist', 28, 1, 10),
]

CIFAR100_MODELS = ['vgg19', 'resnet18', 'resnet50', 'densenet121', 'shufflenet_v2_1_0',
    'mobilenet_v1', 'mobilenet_v2', 'googlenet', 'pre_act_resnet18', 'resnext29_2x64d']

TORCHVISION_MODELS = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'alexnet', 'vgg16', 'squeezenet1_0',
    'densenet161', 'densenet201', 'googlenet', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'mobilenet_v2',
    'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2', 'mnasnet1_0', 'mnasnet0_5',
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'densenet121', 'densenet169',
    'squeezenet1_1', 'q_resnet18', 'q_resnet50', 'q_googlenet', 'q_shufflenet_v2_x0_5', 'q_shufflenet_v2_x1_0',
    'q_mobilenet_v2', 'q_resnext101_32x8d',
]

for model_name in CIFAR100_MODELS:
    CLASSIFICATION_MODEL_TESTS.append((model_name, 'cifar100', 32, 3, 100))

for model_name in TORCHVISION_MODELS:
    CLASSIFICATION_MODEL_TESTS.append((model_name, 'imagenet', 224, 3, 1000))


@pytest.mark.parametrize(
    ('model_name', 'dataset_name', 'input_resolution', 'num_inp_channels', 'target_output_shape'),
    CLASSIFICATION_MODEL_TESTS,
)
def test_classification_model_output_shape(model_name, dataset_name, input_resolution,
    num_inp_channels, target_output_shape):
    model = get_model_by_name(
        model_name=model_name,
        dataset_name=dataset_name,
        pretrained=True,
        progress=False,
        device="cpu",
    )
    y = model(torch.randn(TEST_BATCH_SIZE, num_inp_channels, input_resolution, input_resolution))
    assert y.shape == (TEST_BATCH_SIZE, target_output_shape)


DETECTION_MODEL_TESTS = [
    ('mb1_ssd', 'voc', {}, [(3000, 21), (3000, 4)]),
    ('mb2_ssd_lite', 'voc', {}, [(3000, 21), (3000, 4)]),
    ('mb2_ssd', 'voc', {'num_classes': 2}, [(3000, 21), (3000, 4)]),
    ('resnet18_ssd', 'voc', {'num_classes': 21}, [(8732, 21), (8732, 4)]),
    ('resnet34_ssd', 'voc', {'num_classes': 21}, [(8732, 21), (8732, 4)]),
    ('resnet50_ssd', 'voc', {'num_classes': 21}, [(8732, 21), (8732, 4)]),
]

YOLO_MODELS = ['yolo5s', 'yolo5m', 'yolo5l', 'yolo5x']

for model_name in YOLO_MODELS:
    DETECTION_MODEL_TESTS.append((model_name, 'voc', {'num_classes': 21, 'img_size': 416},
            [(52, 52, 3, 25), (26, 26, 3, 25), (13, 13, 3, 25)]))

YOLO5_6_VOC_MODELS = ['yolo3', 'yolo4s', 'yolo4m', 'yolo4l', 'yolo4l_leaky', 'yolo4x',
    'yolo5_6n', 'yolo5_6s', 'yolo5_6m', 'yolo5_6l', 'yolo5_6x',
    'yolo5_6s_relu', 'yolo5_6m_relu']

YOLO5_6_PERSON_MODELS = ['yolo5_6n', 'yolo5_6s', 'yolo5_6n_relu',
    'yolo5_6s_relu', 'yolo5_6m_relu']

for model_name in YOLO5_6_VOC_MODELS:
    DETECTION_MODEL_TESTS.append((model_name, 'voc', {'num_classes': 21, 'img_size': 416},
            [(3, 52, 52, 25), (3, 26, 26, 25), (3, 13, 13, 25)]))

for model_name in YOLO5_6_PERSON_MODELS:
    DETECTION_MODEL_TESTS.append((model_name, 'person_detection_1', {'num_classes': 1, 'img_size': 320},
            [(3, 40, 40, 6), (3, 20, 20, 6), (3, 10, 10, 6)]))


@pytest.mark.parametrize(
    ('model_name', 'dataset_name', 'datasplit_kwargs', 'output_shapes'),
    DETECTION_MODEL_TESTS
)
def test_detection_model_output_shape(model_name, dataset_name, datasplit_kwargs, output_shapes):
    model = get_model_by_name(
        model_name=model_name,
        dataset_name=dataset_name,
        pretrained=True,
        progress=False,
        device="cpu",
    )
    if model_name in MODEL_NAME_DATASPLIT_FN_ARG_MAP:
        model_name = MODEL_NAME_DATASPLIT_FN_ARG_MAP[model_name]
    train_loader = get_data_splits_by_name(
        data_root=MOCK_VOC_PATH,
        dataset_name=DATASET_NAME_DATASPLIT_FN_ARG_MAP[dataset_name],
        model_name=model_name,
        batch_size=TEST_BATCH_SIZE,
        num_workers=0,
        device="cpu",
        **datasplit_kwargs,
    )["train"]

    if 'yolo' in model_name:
        dataset = train_loader.dataset
        img, _, _, _ = dataset[0]
        y = model(torch.unsqueeze(img, dim=0))
        assert y[0][0].shape == (1, *output_shapes[0])
        assert y[0][1].shape == (1, *output_shapes[1])
        assert y[0][2].shape == (1, *output_shapes[2])
        if y[1] is not None:
            assert y[1][0].shape == (1, *output_shapes[0])
            assert y[1][1].shape == (1, *output_shapes[1])
            assert y[1][2].shape == (1, *output_shapes[2])
    else:
        img, _, _ = next(iter(train_loader))
        model.eval()
        y1, y2 = model(img)
        assert y1.shape == (TEST_BATCH_SIZE, *output_shapes[0])
        assert y2.shape == (TEST_BATCH_SIZE, *output_shapes[1])


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
    if model_name in MODEL_NAME_DATASPLIT_FN_ARG_MAP:
        model_name = MODEL_NAME_DATASPLIT_FN_ARG_MAP[model_name]
    test_loader = get_data_splits_by_name(
        data_root=MOCK_DATASETS_PATH if 'voc' in dataset_name else MOCK_CARVANA_PATH,
        dataset_name=DATASET_NAME_DATASPLIT_FN_ARG_MAP[dataset_name],
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


CUSTOM_NUM_CLASSES = 42

CLASSIFICATION_CREATE_MODEL_TESTS = [
    ('resnet18', 'imagenet16', 224, 3, CUSTOM_NUM_CLASSES),
    ('resnet18', 'imagenet10', 224, 3, CUSTOM_NUM_CLASSES),
    ('resnet18', 'vww', 224, 3, CUSTOM_NUM_CLASSES),
    ('lenet5', 'mnist', 28, 1, CUSTOM_NUM_CLASSES),
    ('wide_resnet50_2', 'imagenet', 224, 3, CUSTOM_NUM_CLASSES),
]

@pytest.mark.parametrize(
    ('model_name', 'dataset_name', 'input_resolution', 'num_inp_channels', 'target_output_shape'),
    CLASSIFICATION_CREATE_MODEL_TESTS,
)
def test_create_classification_model_output_shape(model_name, dataset_name, input_resolution,
    num_inp_channels, target_output_shape):
    model = create_model(
        model_name=model_name,
        pretraining_dataset=dataset_name,
        num_classes=CUSTOM_NUM_CLASSES,
        progress=False,
        device="cpu",
    )
    y = model(torch.randn(TEST_BATCH_SIZE, num_inp_channels, input_resolution, input_resolution))
    assert y.shape == (TEST_BATCH_SIZE, target_output_shape)


DETECTION_CREATE_MODEL_TESTS = [
    ('mb1_ssd', 'voc', {}, [(3000, CUSTOM_NUM_CLASSES+1), (3000, 4)]),
    ('yolo4s', 'voc', {'num_classes': 21, 'img_size': 416},
            [(3, 52, 52, CUSTOM_NUM_CLASSES+5),
             (3, 26, 26, CUSTOM_NUM_CLASSES+5),
             (3, 13, 13, CUSTOM_NUM_CLASSES+5)]),
    ('yolo5_6n', 'voc', {'num_classes': 21, 'img_size': 416},
            [(3, 52, 52, CUSTOM_NUM_CLASSES+5),
             (3, 26, 26, CUSTOM_NUM_CLASSES+5),
             (3, 13, 13, CUSTOM_NUM_CLASSES+5)])
]
@pytest.mark.parametrize(
    ('model_name', 'dataset_name', 'datasplit_kwargs', 'output_shapes'),
    DETECTION_CREATE_MODEL_TESTS,
)
def test_create_detection_model_output_shape(model_name, dataset_name, datasplit_kwargs, output_shapes):
    model = create_model(
        model_name=model_name,
        pretraining_dataset=dataset_name,
        num_classes=CUSTOM_NUM_CLASSES,
        progress=False,
        device="cpu",
    )
    if model_name in MODEL_NAME_DATASPLIT_FN_ARG_MAP:
        model_name = MODEL_NAME_DATASPLIT_FN_ARG_MAP[model_name]
    train_loader = get_data_splits_by_name(
        data_root=MOCK_VOC_PATH,
        dataset_name=DATASET_NAME_DATASPLIT_FN_ARG_MAP[dataset_name],
        model_name=model_name,
        batch_size=TEST_BATCH_SIZE,
        num_workers=0,
        device="cpu",
        **datasplit_kwargs,
    )["train"]

    if 'yolo' in model_name:
        dataset = train_loader.dataset
        img, _, _, _ = dataset[0]
        y = model(torch.unsqueeze(img, dim=0))
        assert y[0][0].shape == (1, *output_shapes[0])
        assert y[0][1].shape == (1, *output_shapes[1])
        assert y[0][2].shape == (1, *output_shapes[2])
        if y[1] is not None:
            assert y[1][0].shape == (1, *output_shapes[0])
            assert y[1][1].shape == (1, *output_shapes[1])
            assert y[1][2].shape == (1, *output_shapes[2])
    else:
        img, _, _ = next(iter(train_loader))
        model.eval()
        y1, y2 = model(img)
        assert y1.shape == (TEST_BATCH_SIZE, *output_shapes[0])
        assert y2.shape == (TEST_BATCH_SIZE, *output_shapes[1])


@pytest.mark.parametrize(
    ('model_name', 'dataset_name', 'datasplit_kwargs', 'output_shape'),
    [
        ('deeplab_mobilenet', 'voc', {'backbone': 'vgg'}, (1, CUSTOM_NUM_CLASSES)),
        ('fcn32', 'voc', {'backbone': 'vgg'}, (1, CUSTOM_NUM_CLASSES)),
    ],
)
def test_create_segmentation_model_output_shape(model_name, dataset_name, datasplit_kwargs, output_shape):
    model = create_model(
        model_name=model_name,
        pretraining_dataset=dataset_name,
        num_classes=CUSTOM_NUM_CLASSES,
        progress=False,
        device="cpu",
    )
    if model_name in MODEL_NAME_DATASPLIT_FN_ARG_MAP:
        model_name = MODEL_NAME_DATASPLIT_FN_ARG_MAP[model_name]
    test_loader = get_data_splits_by_name(
        data_root=MOCK_DATASETS_PATH if 'voc' in dataset_name else MOCK_CARVANA_PATH,
        dataset_name=DATASET_NAME_DATASPLIT_FN_ARG_MAP[dataset_name],
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
