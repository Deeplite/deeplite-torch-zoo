<p align="center">
  <img src="https://docs.deeplite.ai/neutrino/_static/content/deeplite-logo-color.png" />
</p>

[![Build Status](https://travis-ci.com/Deeplite/deeplite-torch-zoo.svg?token=kodd5rKMpjxQDqRCxwiV&branch=master)](https://travis-ci.com/Deeplite/deeplite-torch-zoo) [![codecov](https://codecov.io/gh/Deeplite/deeplite-torch-zoo/branch/master/graph/badge.svg?token=AVTp3PW5UP)](https://codecov.io/gh/Deeplite/deeplite-torch-zoo)

# Deeplite Torch Zoo

The ``deeplite-torch-zoo`` package is a collection of popular CNN model architectures and benchmark datasets for PyTorch. The models are grouped under different datasets and different task types such as classification, object detection, and semantic segmentation. The primary aim of ``deeplite-torch-zoo`` is to booststrap applications by starting with the most suitable pretrained models for a given task. In addition, the pretrained models from ``deeplite-torch-zoo`` can be used as a good starting point for optimizing model architectures using our [neutrino_engine](https://docs.deeplite.ai/neutrino/index.html)

* [Installation](#Installation)
    * [Install using pip](#Install-using-pip-release-version)
    * [Install from source](##install-from-source-development-version)
    * [Install in Dev mode](#Install-in-dev-mode)

* [How to Use](#How-to-Use)
    * [Loading Datasets](#Loading-Datasets)
        * [Classification Datasets](#Classification-Datasets)
        * [Object Detection Datasets](#Object-Detection-Datasets)
    * [Loading Models](#Loading-Models)
        * [Classification Models](#Classification-Models)
        * [Object Detection Models](#Object-Detection-Models)

* [Available Models](#Available-Models)
* [Available Datasets](#Available-Datasets)
* [Benchmark Results](#Benchmark-Results)
* [Contribute a Model/Dataset to the Zoo](#Contribute-a-Model/Dataset-to-the-Zoo)


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

The ``deeplite-torch-zoo`` is collection of benchmark computer vision datasets and pretrained models. There are three primary wrapper functions to load datasets, models and evaluation functions: ``get_data_splits_by_name``, ``get_model_by_name``, ``get_eval_function`` which can be imported as

```{.python}
from deeplite_torch_zoo import get_data_splits_by_name
from deeplite_torch_zoo import get_model_by_name
from deeplite_torch_zoo import get_eval_function
```

## Loading Datasets

The loaded datasets are available as a dictionary of the following format: ``{'train': train_dataloder, 'test': test_dataloader}``. The `train_dataloder` and `test_dataloader` are objects of ``torch.utils.data.DataLoader``.

### Classification Datasets


```{.python}
    data_splits = get_data_splits_by_name(
        dataset_name="cifar100", model_name="resnet18", batch_size=128
    )
```
The list of all available classification datasets can be found [here].

### Object Detection Datasets

The following sample code loads the [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) dataset. ``train`` contains the data loader for train sets for `VOC2007` and/or `VOC2012`. If both datasets are provided it concatenates both `VOC2007` and `VOC2012` train sets. Otherwise, it returns the train set for the provided dataset. 'test' contains dataloader (always with ``batch_size=1``) for test set based on `VOC2007`. You also need to provide the model name as well.

```{.python}
data_splits = get_data_splits_by_name(
        data_root=PATH_TO_VOCdevkit,
        dataset_name="voc",
        model_name="yolo3",
        batch_size=BATCH_SIZE,
    )
```

> **_NOTE:_**  As it can be observed the data_loaders are provided based on the corresponding model (`model_name`). Different object detection models consider inputs/outputs in different formats, and thus the our `data_splits` are formatted according to the needs of the model (e.g. for SSD or YOLO detection models).

## Loading Models

Models are provided with weights pretrained on specific datasets. Thus, one could load a model ``X`` pretrained on dataset ``Y``, for getting the appropriate weights.

### Classification Models

```{.python}
    model = get_model_by_name(
        model_name="resnet18",
        dataset_name="cifar100",
        pretrained=True, # or False, if pretrained weights are not required
        progress=False, # or True, if a progressbar is required
        device="cpu", # or "gpu"
    )
```
The list of all available classification models can be found [here].

### Object Detection Models

```{.python}
    model = get_model_by_name(
        model_name="vgg16_ssd",
        dataset_name="voc",
        pretrained=True, # or False, if pretrained weights are not required
        progress=False, # or True, if a progressbar is required
    )
```

To evaluate a model, the following style of code could be used,

```{.python}
    test_loader = data_splits["test"]
    eval_function = get_eval_function(dataset_name="voc", model_name="vgg16_ssd")
    APs = eval_function(model, test_loader)
```

### Creating a custom model based on existing architecture

One could create a model with a custom number of classes, while loading the pretrained weights from one of the pretrained models available in the zoo for the modules where weight reusage is possible. An example below creates a ``yolo5_6m`` model with 8 output classes, with weights loaded from a COCO checkpoint (except for the layers whise shape depends on the number of classes):

```{.python}
    from deeplite_torch_zoo import create_model

    model = create_model(
        model_name="yolo5_6m",
        pretraining_dataset="coco",
        num_classes=8,
        pretrained=True, # or False, if pretrained weights are not required
        progress=False, # or True, if a progressbar is required
    )
```

# Available Models

There is an important utility function ``list_models`` which can be imported as
```{.python}
from deeplite_torch_zoo import list_models
```
This utility will help in listing all available pretrained models or datasets.

For instance ``list_models("yolo5")`` will provide the following result. Similar results can be obtained using ``list_models("yo")``.

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

# Available Datasets
 - Object detection: VOC, COCO, WiderFace, Person Detection (subsampled COCO)
 - Semantic Segmentation: Carvana, VOC

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
- The implementation of mb-ssd models: [qfgaohao/pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd)
- The implementation of resnet-ssd: [Nvidia-SSD](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD)
- The implementation of yolov5: [ultralytics/yolov5](https://github.com/ultralytics/yolov5)

### Segmentation
- The implementation of deeplab: [pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)
- The implementation of unet_scse: [nyoki-mtl/pytorch-segmentation](https://github.com/nyoki-mtl/pytorch-segmentation)
- The implementation of fcn: [wkentaro/pytorch-fcn](https://github.com/wkentaro/pytorch-fcn)
- The implementation of Unet: [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)

### Classification
- The implementation of models on CIFAR100 dataset: [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)
- The implementation of Mobilenetv1 model on VWW dataset: [qfgaohao/pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd)
- The implementation of Mobilenetv3 model on VWW dataset: [d-li14/mobilenetv3.pytorch](https://github.com/d-li14/mobilenetv3.pytorch)
