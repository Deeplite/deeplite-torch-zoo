import pytest

from deeplite_torch_zoo import get_model
from deeplite_torch_zoo.trainer import Detector

TEST_MODELS = [
    ('yolo7n', ),
    ('yolo_fdresnet18x0.25', ),
    ('yolo_timm_tinynet_e', ),
    ('yolo6s-d33w25', ),
    ('yolonas_s', ),
]


@pytest.mark.parametrize(
    ('model_name', ),
    TEST_MODELS,
)
def test_detection_trainer(model_name):
    model = Detector(model_name=model_name)
    model.train(data='coco8.yaml', epochs=1)


@pytest.mark.parametrize(
    ('model_name', ),
    TEST_MODELS,
)
def test_detection_trainer_torch_model(model_name):
    torch_model = get_model(
        model_name=model_name,
        dataset_name='coco',
        pretrained=False,
        custom_head='yolo8',
    )
    model = Detector(torch_model=torch_model)
    model.train(data='coco8.yaml', epochs=1)
