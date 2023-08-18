<div align="center">

  ![logo](https://docs.deeplite.ai/neutrino/_static/content/deeplite-logo-color.png)

  **🚀 Deeplite Torch Zoo 🚀 is a collection of state-of-the-art efficient
  computer vision models for embedded applications in [PyTorch](https://pytorch.org/).**

  [![Build Status](https://travis-ci.com/Deeplite/deeplite-torch-zoo.svg?token=kodd5rKMpjxQDqRCxwiV&branch=master)](https://travis-ci.com/Deeplite/deeplite-torch-zoo) [![codecov](https://codecov.io/gh/Deeplite/deeplite-torch-zoo/branch/master/graph/badge.svg?token=AVTp3PW5UP)](https://codecov.io/gh/Deeplite/deeplite-torch-zoo)

</div>

**For information on YOLOBench, click [here](results/yolobench).**

The main features of this library are:

 - High-level API to create models, dataloaders, and evaluation functions
 - Single interface for SOTA classification models:
    - [timm models](https://github.com/huggingface/pytorch-image-models/),
    - [pytorchcv models](https://github.com/osmr/imgclsmob/tree/master/pytorch),
    - other SOTA efficient models (EdgeViT, FasterNet, GhostNetV2, MobileOne)
 - Single interface for SOTA YOLO detectors (compatible with [Ultralytics training](https://github.com/ultralytics/ultralytics)):
    - YOLOv3, v4, v5, v6-3.0, v7, v8
    - YOLO with timm backbones
    - [other experimental configs](https://github.com/Deeplite/deeplite-torch-zoo/tree/develop/deeplite_torch_zoo/src/object_detection/yolov5/configs)

### 📋 Table of content
 1. [Quick start](#start)
 2. [Installation](#installation)
 3. [Training scripts](#training-scripts)
 7. [Contributing](#contributing)
 9. [Credit](#credit)


### ⏳ Quick start <a name="start"></a>

#### Create a classification model

```python
from deeplite_torch_zoo import get_model, list_models

model = get_model(
    model_name='edgevit_xs',        # model names for imagenet available via `list_models('imagenet')`
    dataset_name='imagenet',        # dataset name, since resnet18 is different for e.g. imagenet and cifar100
    pretrained=False,               # if True, will try to load a pre-trained checkpoint
)

# creating a model with 42 classes for transfer learning:

model = get_model(
    model_name='fasternet_t0',        # model names for imagenet available via `list_models('imagenet')`
    num_classes=42,                   # number of classes for transfer learning
    dataset_name='imagenet',   # take weights from checkpoint pre-trained on this dataset
    pretrained=False,                 # if True, will try to load all weights with matching tensor shapes
)
```

#### Create an object detection model

```python
from deeplite_torch_zoo import get_model

model = get_model(
    model_name='yolo4n',        # creates a YOLOv4n model on COCO
    dataset_name='coco',        # (`n` corresponds to width factor 0.25, depth factor 0.33)
    pretrained=False,           # if True, will try to load a pre-trained checkpoint
)

# one could create a YOLO model with timm backbone,
# PAN neck and YOLOv8 decoupled anchor-free head like this:

model = get_model(
    model_name='yolo_timm_fbnetv3_d',    # creates a YOLO with FBNetV3-d backbone from timm
    dataset_name='coco',                 #
    pretrained=False,                    # if True, will try to load a pre-trained checkpoint
    custom_head='v8',                    # will replace default detection head
                                         # with YOLOv8 detection head
)
```

#### Create PyTorch dataloaders

```python
from deeplite_torch_zoo import get_dataloaders

dataloaders = get_dataloaders(
    data_root='./',                      # folder with data, will be used for download
    dataset_name='imagewoof',            # datasets to if applicable,
    num_workers=8,                       # number of dataloader workers
    batch_size=64,                       # dataloader batch size (train and test)
)

# dataloaders['train'] -> train dataloader
# dataloaders['test'] -> test dataloader
#
# see below for the list of supported datasets
```

The list of supported datasets is available for [classification](https://github.com/Deeplite/deeplite-torch-zoo/blob/develop/docs/CLASSIFICATION.md) and [object detection](https://github.com/Deeplite/deeplite-torch-zoo/blob/develop/docs/OBJECT_DETECTION.md).

#### Creating an evaluation function

```python
from deeplite_torch_zoo import get_eval_function

eval_function = get_eval_function(
    model_name='yolo8s',
    dataset_name='voc',
)

# required arg signature is fixed for all eval functions
metrics = eval_function(model, test_dataloader)
```

#### (Experimental) Training with patched Ultralytics trainer

```python
from deeplite_torch_zoo.trainer import Detector

model = Detector(model_name='yolo7n')        # will create a wrapper around YOLOv7n model
                                             # (YOLOv7n model with YOLOv8 detection head)
model.train(data='VOC.yaml', epochs=100)     # same arguments as Ultralytics trainer
```

### 🛠 Installation <a name="installation"></a>
PyPI version:
```bash
$ pip install deeplite-torch-zoo
````
Latest version from source:
```bash
$ pip install git+https://github.com/Deeplite/deeplite-torch-zoo.git
````

### 💪 Training scripts <a name="training-scripts"></a>

We provide several training scripts as an example of how `deeplite-torch-zoo` can be integrated into existing training pipelines:


- [modified timm ImageNet script](https://github.com/Deeplite/deeplite-torch-zoo/tree/develop/training_scripts/classification/imagenet)

  - support for Knowledge Distillation
  - training recipes provides (A1, A2, A3, [USI](https://github.com/Alibaba-MIIL/Solving_ImageNet), etc.)
- [modfied Ultralytics classification fine-tuning script](https://github.com/Deeplite/deeplite-torch-zoo/tree/develop/training_scripts/classification/ultralytics)
- [modfied Ultralytics YOLOv5 object detector training script](https://github.com/Deeplite/deeplite-torch-zoo/tree/develop/training_scripts/object_detection)


### 🤝 Contributing <a name="contributing"></a>

We always welcome community contributions to expand the scope of `deeplite-torch-zoo` and also to have additional new models and datasets. Please refer to the [documentation](https://docs.deeplite.ai/neutrino/zoo.html#contribute-a-model-dataset-to-the-zoo) for the detailed steps on how to add a model and dataset. In general, we follow the `fork-and-pull` Git workflow.

 1. **Fork** the repo on GitHub
 2. **Clone** the project to your own machine
 3. **Commit** changes to your own branch
 4. **Push** your work back up to your fork
 5. Submit a **Pull request** so that we can review your changes

NOTE: Be sure to merge the latest from "upstream" before making a pull request!

## 🙏 Credit <a name="credit"></a>

<details>

  <summary>Repositories used to build Deeplite Torch Zoo</summary>

### Object Detection
- YOLOv3 implementation: [ultralytics/yolov3](https://github.com/ultralytics/yolov3)
- YOLOv5 implementation: [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- flexible-yolov5 implementation: [Bobo-y/flexible-yolov5](https://github.com/Bobo-y/flexible-yolov5)
- YOLOv8 implementation: [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- YOLOv7 implementation: [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
- YOLOX implementation: [iscyy/yoloair](https://github.com/iscyy/yoloair)
- [westerndigitalcorporation/YOLOv3-in-PyTorch](https://github.com/westerndigitalcorporation/YOLOv3-in-PyTorch)

### Segmentation
- The implementation of deeplab: [pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)
- The implementation of unet_scse: [nyoki-mtl/pytorch-segmentation](https://github.com/nyoki-mtl/pytorch-segmentation)
- The implementation of fcn: [wkentaro/pytorch-fcn](https://github.com/wkentaro/pytorch-fcn)
- The implementation of Unet: [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)

### Classification
- The implementation of models on CIFAR100 dataset: [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)
- The implementation of Mobilenetv1 model on VWW dataset: [qfgaohao/pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd)
- The implementation of Mobilenetv3 model on VWW dataset: [d-li14/mobilenetv3.pytorch](https://github.com/d-li14/mobilenetv3.pytorch)

### DNN building block implementations
- [d-li14/mobilenetv2.pytorch](https://github.com/d-li14/mobilenetv2.pytorch)
- [d-li14/efficientnetv2.pytorch](https://github.com/d-li14/efficientnetv2.pytorch)
- [apple/ml-mobileone](https://github.com/apple/ml-mobileone)
- [osmr/imgclsmob](https://github.com/osmr/imgclsmob)
- [huggingface/pytorch-image-models](https://github1s.com/huggingface/pytorch-image-models)
- [moskomule/senet.pytorch](https://github.com/moskomule/senet.pytorch)
- [DingXiaoH/RepLKNet-pytorch](https://github.com/DingXiaoH/RepLKNet-pytorch)
- [huawei-noah/Efficient-AI-Backbones](https://github.com/huawei-noah/Efficient-AI-Backbones)


### Misc
- torchvision dataset implementations: [pytorch/vision](https://github.com/pytorch/vision)
- MLP implementation: [aaron-xichen/pytorch-playground](https://github.com/aaron-xichen/pytorch-playground)
- AutoAugment implementation: [DeepVoltaire/AutoAugment](https://github.com/DeepVoltaire/AutoAugment)
- Cutout implementation: [uoguelph-mlrg/Cutout](https://github.com/uoguelph-mlrg/Cutout)
- Robustness measurement image distortions: [hendrycks/robustness](https://github.com/hendrycks/robustness)
- Registry implementation: [openvinotoolkit/openvino/tools/pot](https://github.com/openvinotoolkit/openvino/blob/master/tools/pot)

</details>
