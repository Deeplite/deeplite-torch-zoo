import urllib

import pandas as pd

from deeplite_torch_zoo import get_model
from deeplite_torch_zoo.trainer import Detector


DATA_URL = 'https://raw.githubusercontent.com/Deeplite/deeplite-torch-zoo/develop/results/yolobench/'
df = pd.read_csv(urllib.parse.urljoin(DATA_URL, 'VOC_scratch_100epochs.csv'))

for model_name in df.model_name:
    torch_model = get_model(
        model_name=model_name.split('_')[0],
        dataset_name='coco',
        num_classes=80,
        pretrained=False,
        custom_head='yolo8',
    )
    model = Detector(torch_model=torch_model)
    model.export(
        format='onnx',
        model_name=model_name,
        opset=11,
        simplify=True,
        imgsz=int(model_name.split('_')[1]),
    )
