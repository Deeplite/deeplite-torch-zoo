import pytest

from deeplite_torch_zoo.trainer import Detector


@pytest.mark.parametrize(
    ('model_name', ),
    [
        ('yolo7n', ),
        ('yolo_fdresnet18x0.25', ),
        ('yolo_timm_tinynet_e', ),
        ('yolo6s-d33w25', ),
    ],
)
def test_detection_trainer(model_name):
    model = Detector(model_name=model_name)
    model.train(data='coco8.yaml', epochs=1)
