import pytest 

from deeplite_torch_zoo import Detector


@pytest.mark.parametrize(
    ('model_name', ),
    [('yolo7n', ),],
)
def test_detection_trainer(model_name):
    model = Detector(model_name=model_name)
    model.train(data='coco8.yaml', epochs=1)
