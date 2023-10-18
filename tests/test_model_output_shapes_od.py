import pytest

from deeplite_torch_zoo import get_dataloaders, get_model, list_models_by_dataset
from deeplite_torch_zoo.api.models.object_detection.yolo import YOLO_CONFIGS


TEST_BATCH_SIZE = 4
TEST_NUM_CLASSES = 42
COCO_NUM_CLASSES = 80

DETECTION_MODEL_TESTS = []

for model_key in YOLO_CONFIGS:
    DETECTION_MODEL_TESTS.append((f'{model_key}t', 'coco8', {'image_size': 480},
        [(3, 64, 64), (3, 32, 32), (3, 16, 16)], False, False))

for model_name in list_models_by_dataset('coco', with_checkpoint=True):
    DETECTION_MODEL_TESTS.append((model_name, 'coco8', {'image_size': 480},
        [(3, 64, 64), (3, 32, 32), (3, 16, 16)], True, True))


@pytest.mark.parametrize(
    ('model_name', 'dataset_name', 'dataloader_kwargs',
     'output_shapes', 'download_checkpoint', 'check_shape'),
    DETECTION_MODEL_TESTS
)
def test_detection_model_output_shape(
    model_name,
    dataset_name,
    dataloader_kwargs,
    output_shapes,
    download_checkpoint,
    check_shape,
    ):
    model = get_model(
        model_name=model_name,
        dataset_name='coco',
        pretrained=download_checkpoint,
    )
    dataloader = get_dataloaders(
        data_root='./',
        dataset_name=dataset_name,
        batch_size=TEST_BATCH_SIZE,
        num_workers=0,
        **dataloader_kwargs,
    )['test']
    img, *_ = next(iter(dataloader))
    img = img / 255
    y = model(img)
    y[0].sum().backward()

    if check_shape:
        assert y[0].shape == (4, *output_shapes[0], COCO_NUM_CLASSES + 5)
        assert y[1].shape == (4, *output_shapes[1], COCO_NUM_CLASSES + 5)
        assert y[2].shape == (4, *output_shapes[2], COCO_NUM_CLASSES + 5)


@pytest.mark.parametrize(
    ('model_name', 'dataset_name', 'dataloader_kwargs',
     'output_shapes', 'download_checkpoint', 'check_shape'),
    DETECTION_MODEL_TESTS
)
def test_detection_model_output_shape_arbitrary_num_clases(
    model_name,
    dataset_name,
    dataloader_kwargs,
    output_shapes,
    download_checkpoint,
    check_shape,
    ):
    model = get_model(
        model_name=model_name,
        num_classes=TEST_NUM_CLASSES,
        dataset_name='coco',
        pretrained=download_checkpoint,
    )
    dataloader = get_dataloaders(
        data_root='./',
        dataset_name=dataset_name,
        batch_size=TEST_BATCH_SIZE,
        num_workers=0,
        **dataloader_kwargs,
    )['test']
    img, *_ = next(iter(dataloader))
    img = img / 255
    y = model(img)
    y[0].sum().backward()

    if check_shape:
        assert y[0].shape == (4, *output_shapes[0], TEST_NUM_CLASSES + 5)
        assert y[1].shape == (4, *output_shapes[1], TEST_NUM_CLASSES + 5)
        assert y[2].shape == (4, *output_shapes[2], TEST_NUM_CLASSES + 5)
