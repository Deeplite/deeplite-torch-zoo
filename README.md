
<p align="center">
  <img src="https://docs.deeplite.ai/neutrino/_static/content/deeplite-logo-color.png" />
</p>

[![Build Status](https://travis-ci.com/Deeplite/deeplite-torch-zoo.svg?token=kodd5rKMpjxQDqRCxwiV&branch=master)](https://travis-ci.com/Deeplite/deeplite-torch-zoo) [![codecov](https://codecov.io/gh/Deeplite/deeplite-torch-zoo/branch/master/graph/badge.svg?token=AVTp3PW5UP)](https://codecov.io/gh/Deeplite/deeplite-torch-zoo)

# 🚀 Deeplite Torch Zoo 🚀

The ``deeplite-torch-zoo`` package is a collection of popular (pretrained) CNN model architectures and benchmark datasets for PyTorch. The models are grouped under different datasets and different task types such as classification, object detection, and semantic segmentation. The primary aim of ``deeplite-torch-zoo`` is to booststrap applications by starting with the most suitable pretrained models for a given task. In addition, the pretrained models from ``deeplite-torch-zoo`` can be used as a good starting point for optimizing model architectures using our [neutrino_engine](https://docs.deeplite.ai/neutrino/index.html)

- [Deeplite Torch Zoo](#deeplite-torch-zoo)
- [Installation](#installation)
  - [Install using pip (release version)](#install-using-pip-release-version)
  - [Install from source (development version)](#install-from-source-development-version)
  - [Install in dev mode](#install-in-dev-mode)
- [How to Use](#how-to-use)
  - [Loading Datasets](#loading-datasets)
    - [Classification Datasets](#classification-datasets)
    - [Object Detection Datasets](#object-detection-datasets)
  - [Loading and Creating Models](#loading-and-creating-models)
    - [Classification Models](#classification-models)
    - [Object Detection Models](#object-detection-models)
  - [Creating an evaluation function](#creating-an-evaluation-function)
- [Available Models](#available-models)
- [Train on Custom Dataset](#train-on-custom-dataset)
- [Benchmark Results](#benchmark-results)
- [Contribute a Model/Dataset to the Zoo](#contribute-a-modeldataset-to-the-zoo)
  - [Credit](#credit)
    - [Object Detection](#object-detection)
    - [Segmentation](#segmentation)
    - [Classification](#classification)
    - [DNN building block implementations](#dnn-building-block-implementations)
    - [Misc](#misc)


# Installation

## Install using pip (release version)


Use following command to install the package from our internal PyPI repository.

```
    $ pip install --upgrade pip
    $ pip install deeplite-torch-zoo
```

## Install from source (development version)

```
    $ git clone https://github.com/Deeplite/deeplite-torch-zoo.git
    $ pip install .
```

## Install in dev mode

```
    $ git clone https://github.com/Deeplite/deeplite-torch-zoo.git
    $ pip install -e .
    $ pip install -r requirements-test.txt
```

To test the installation, one can run the basic tests using `pytest` command in the root folder.

# How to Use

The ``deeplite-torch-zoo`` is collection of benchmark computer vision datasets and pretrained models. There are four primary wrapper functions to load datasets, models and evaluation functions: ``get_dataloaders``, ``get_model``, ``get_eval_function`` and ``create_model`` which can be imported as

```{.python}
from deeplite_torch_zoo import get_dataloaders
from deeplite_torch_zoo import get_model
from deeplite_torch_zoo import get_eval_function
from deeplite_torch_zoo import create_model
```

## Loading Datasets

The loaded datasets are available as a dictionary of the following format: ``{'train': train_dataloder, 'test': test_dataloader}``. The `train_dataloder` and `test_dataloader` are objects of type ``torch.utils.data.DataLoader``.

### Classification Datasets


```{.python}
    data_splits = get_dataloaders(
        data_root="./", dataset_name="cifar100", model_name="resnet18", batch_size=128
    )
```
The list of all available classification datasets can be found [here](docs/CLASSIFICATION.md/#datasets). Please note that it is always necessary to pass the model name upon the creation of dataloader because the dataset class logic might depend on the model type.

### Object Detection Datasets

The following sample code loads the [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) dataset. ``train`` contains the data loader for the trainval data splits of the `VOC2007` and/or `VOC2012`. If both datasets are provided it concatenates both `VOC2007` and `VOC2012` train sets. Otherwise, it returns the train set for the provided dataset. 'test' contains dataloader (always with ``batch_size=1``) for the test split of `VOC2007`. You also need to provide the model name to instantiate the dataloaders.

```{.python}
data_splits = get_dataloaders(
        data_root=PATH_TO_VOCdevkit,
        dataset_name="voc",
        model_name="yolo3",
        batch_size=BATCH_SIZE,
    )
```
The list of all available object detection datasets can be found [here](docs/OBJECT_DETECTION.md/#datasets).

> **_NOTE:_**  As it can be observed the data_loaders are provided based on the corresponding model (`model_name`). Different object detection models consider inputs/outputs in different formats, and thus the our `data_splits` are formatted according to the needs of the model (e.g. for SSD or YOLO detection models).

## Loading and Creating Models

Models are generally provided with weights pretrained on specific datasets. One would load a model ``X`` pretrained on a dataset ``Y`` to get the appropriate weights for the task ``Y``. The ``get_model`` could used for this purpose. There is also an option to create a new model with an arbitrary number of categories for the downstream tasl and load the weights from another dataset for transfer learning (e.g. to load ``COCO`` weights to train a model on the ``VOC`` dataset). The ``create_model`` method should be generally used for that. Note that ``get_model`` always returns a fully-trained model for the specified task, this method thus does not allow specifying a custom number of classes.

### Classification Models

To get a pretrained classification model one could use

```{.python}
    model = get_model(
        model_name="resnet18",
        dataset_name="cifar100",
        pretrained=True, # or False, if pretrained weights are not required
    )
```

To create a new model with ImageNet weights and a custom number of classes one could use

```{.python}
    model = create_model(
        model_name="resnet18",
        pretraining_dataset="imagenet",
        num_classes=42,
        pretrained=True, # or False, if pretrained weights are not required
    )
```

This method would load the ImageNet-pretrained weights to all the modules of the model where one could match the shape of the weight tensors (i.e. all the layers except the last fully-connected one in the above case).

The list of all available classification models can be found [here](docs/CLASSIFICATION.md/#complete-list-of-models-and-datasets).

### Object Detection Models

```{.python}
    model = get_model(
        model_name="yolo4s",
        dataset_name="voc",
        pretrained=True, # or False, if pretrained weights are not required
    )
```

Likewise, to create a object detection model with an arbitrary number of classes

```{.python}
    model = get_model(
        model_name="yolo4s",
        num_classes=5,
        dataset_name="coco",
        pretrained=True, # or False, if pretrained weights are not required
    )
```

The list of all available Object Detection models can be found [here](docs/OBJECT_DETECTION.md/#complete-list-of-models-and-datasets).

## Creating an evaluation function

To create an evaluation fuction for the given model and dataset one could call ``get_eval_function`` passing the ``model_name`` and ``dataset_name`` arguments:

```{.python}
    eval_fn = get_eval_function(
        model_name="resnet50",
        dataset_name="imagenet",
    )
```

The returned evaluation function is a Python callable that takes two arguments: a PyTorch model object and a PyTorch dataloader object (logically corresponding to the test split dataloader) and returns a dictionary with metric names as keys and their corresponding values.


# Available Models

There is an useful utility function ``list_models`` which can be imported as
```{.python}
from deeplite_torch_zoo import list_models
```
This utility will help in listing available pretrained models or datasets.

For instance ``list_models("yolo5")`` will provide the list of available pretrained models that contain ``yolo5`` in their model names. Similar results e.g. can be obtained using ``list_models("yo")``. Filtering models by the corresponding task type is also possible by passing the string of the task type with the ``task_type_filter`` argument (the following task types are available: ``classification``, ``object_detection``, ``semantic_segmentation``).

```
    +------------------+------------------------------------+
    | Available models |          Source datasets           |
    +==================+====================================+
    | yolo5_6l         | voc                                |
    +------------------+------------------------------------+
    | yolo5_6m         | coco, voc                          |
    +------------------+------------------------------------+
    | yolo5_6m_relu    | person_detection, voc              |
    +------------------+------------------------------------+
    | yolo5_6ma        | coco                               |
    +------------------+------------------------------------+
    | yolo5_6n         | coco, person_detection, voc, voc07 |
    +------------------+------------------------------------+
    | yolo5_6n_hswish  | coco                               |
    +------------------+------------------------------------+
    | yolo5_6n_relu    | coco, person_detection, voc        |
    +------------------+------------------------------------+
    | yolo5_6s         | coco, person_detection, voc, voc07 |
    +------------------+------------------------------------+
    | yolo5_6s_relu    | person_detection, voc              |
    +------------------+------------------------------------+
    | yolo5_6sa        | coco, person_detection             |
    +------------------+------------------------------------+
    | yolo5_6x         | voc                                |
    +------------------+------------------------------------+
```


# Train on Custom Dataset

One could refer to the example [training scripts](../training_scripts/) to see how the zoo could be integrated into differen training pipelines. For more details please see

- [Training a Classification Model](docs/CLASSIFICATION.md/#training-on-custom-dataset)
- [Training an Object Detection Model](docs/OBJECT_DETECTION.md/#training-on-custom-dataset)


# Benchmark Results

Please refer to our [documentation](https://docs.deeplite.ai/neutrino/zoo.html#zoo-benchmark-results) for the detailed performance metrics of the pretrained models available in the ``deeplite-torch-zoo``. After downloading a model, please evaluate the model using [deeplite-profiler](https://docs.deeplite.ai/neutrino/profiler.html) to verify the performance metric values. However, one may see different numbers for the execution time as the target hardware and/or the load on the system may impact it.

# Contribute a Model/Dataset to the Zoo

> **_NOTE:_**  If you looking for an SDK documentation, please head over [here](https://deeplite.github.io/deeplite-torch-zoo/).

We always welcome community contributions to expand the scope of `deeplite-torch-zoo` and also to have additional new models and datasets. Please refer to the [documentation](https://docs.deeplite.ai/neutrino/zoo.html#contribute-a-model-dataset-to-the-zoo) for the detailed steps on how to add a model and dataset. In general, we follow the `fork-and-pull` Git workflow.

 1. **Fork** the repo on GitHub
 2. **Clone** the project to your own machine
 3. **Commit** changes to your own branch
 4. **Push** your work back up to your fork
 5. Submit a **Pull request** so that we can review your changes

NOTE: Be sure to merge the latest from "upstream" before making a pull request!


## Credit

### Object Detection
- The implementation of yolov3-voc: [Peterisfar/YOLOV3](https://github.com/Peterisfar/YOLOV3/)
- The implementation of yolov3: [ultralytics/yolov3](https://github.com/ultralytics/yolov3)
- The implementation of yolov5: [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- The implementation of flexible-yolov5: [Bobo-y/flexible-yolov5](https://github.com/Bobo-y/flexible-yolov5)
- The implementation of yolov7: [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
- The implementation of yolox: [iscyy/yoloair](https://github.com/iscyy/yoloair)
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
- mAP metric calculation code: [bes-dev/mean_average_precision](https://github.com/bes-dev/mean_average_precision)
- torchvision dataset implementations: [pytorch/vision](https://github.com/pytorch/vision)
- MLP implementation: [aaron-xichen/pytorch-playground](https://github.com/aaron-xichen/pytorch-playground)
- AutoAugment implementation: [DeepVoltaire/AutoAugment](https://github.com/DeepVoltaire/AutoAugment)
- Cutout implementation: [uoguelph-mlrg/Cutout](https://github.com/uoguelph-mlrg/Cutout)
- Robustness measurement image distortions: [hendrycks/robustness](https://github.com/hendrycks/robustness)
- Registry implementation: [openvinotoolkit/openvino/tools/pot](https://github.com/openvinotoolkit/openvino/blob/master/tools/pot)
