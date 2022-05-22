<p align="center">
  <img src="https://docs.deeplite.ai/neutrino/_static/content/deeplite-logo-color.png" />
</p>

[![Build Status](https://travis-ci.com/Deeplite/deeplite-torch-zoo.svg?token=kodd5rKMpjxQDqRCxwiV&branch=master)](https://travis-ci.com/Deeplite/deeplite-torch-zoo) [![codecov](https://codecov.io/gh/Deeplite/deeplite-torch-zoo/branch/master/graph/badge.svg?token=AVTp3PW5UP)](https://codecov.io/gh/Deeplite/deeplite-torch-zoo)

# Deeplite Torch Zoo

The ``deeplite-torch-zoo`` package is a collection of popular CNN model architectures and benchmark datasets for PyTorch. The models are grouped under different datasets and different task types such as classification, object detection, and semantic segmentation. The primary aim of ``deeplite-torch-zoo`` is to booststrap applications by starting with the most suitable pretrained models for a given task. In addition, the pretrained models from ``deeplite-torch-zoo`` can be used as a good starting point for optimizing model architectures using our [neutrino_engine](https://docs.deeplite.ai/neutrino/index.html)

* [Installation](#Installation)
    * [Install using pip](#Install-using-pip)
    * [Install from source](#Install-from-source)
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
from deeplite_torch_zoo import get_data_splits_by_name, get_model_by_name, get_eval_function
```

## Loading Datasets

The loaded datasets are available as a dictionary of the following format: ``{'train': train_dataloder, 'test': test_dataloader}``. The `train_dataloder` and `test_dataloader` are objects of ``torch.utils.data.DataLoader``.

### Classification Datasets


```{.python}
    # Example: DATASET_NAME = "cifar100", BATCH_SIZE = 128
    data_splits = get_data_splits_by_name(
        dataset_name=DATASET_NAME, batch_size=BATCH_SIZE
    )
```

### Object Detection Datasets

The following sample code loads `PASCAL VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ dataset. ``train`` contains data loader for train sets for `VOC2007` and/or `VOC2012`. If both datasets are provided it concatenates both `VOC2007` and `VOC2012` train sets. Otherwise, it returns the train set for the provided dataset. 'test' contains dataloader (always with ``batch_size=1``) for test set based on `VOC2007`. You also need to provide the model type as well.

```{.python}
data_splits = get_data_splits_by_name(
        data_root=PATH_TO_VOCdevkit,
        dataset_name="voc",
        model_name="vgg16_ssd",
        batch_size=BATCH_SIZE,
    )
```

> **_NOTE:_**  As it can be observed the data_loaders are provided based on the corresponding model (`model_name`). Different object detection models consider inputs/outputs in different formats, and thus the our `data_splits` are formatted according to the needs of the model.

## Loading Models

Models are provided with weights pretrained on specific datasets. Thus, one could load a model ``X`` pretrained on dataset ``Y``, for getting the appropriate weights.

### Classification Models

```{.python}
    model = get_model_by_name(
        model_name=MODEL_NAME, # example: "resnet18"
        dataset_name=DATASET_NAME, # example: "cifar100"
        pretrained=True, # or False, if pretrained weights are not required
        progress=False, # or True, if a progressbar is required
        device="cpu", # or "gpu"
    )
```

### Object Detection Models

```{.python}
    model = get_model_by_name(
        model_name=MODEL_NAME, # example: "vgg16_ssd"
        dataset_name=DATASET_NAME, # example: "voc_20"
        pretrained=True, # or False, if pretrained weights are not required
        progress=False, # or True, if a progressbar is required
    )
```

To evaluate a model, the following style of code could be used,

```{.python}
    test_loader = data_splits["test"]
    APs = vgg16_ssd_eval_func(model, test_loader)
```




# Available Models

There is an important utility function ``list_models`` which can be imported as
```{.python}
from deeplite_torch_zoo import list_models
```
 This utility will help in listing all available pretrained models or datasets.

For instance ``list_models("yolo3")`` will provide the following result. Similar results can be obtained using ``list_models("yo")``.

```
    yolo3
    yolo3_voc_1
    yolo3_voc_2
    yolo3_voc_6
    yolo3_voc_20
    yolo3_lisa_11
```



# Available Datasets

| # | Dataset (dataset_name) | Training Instances | Test Instances       | Resolution | Comments                               |
| --|------------------------| -------------------|----------------------|----------- |----------------------------------------|
| 1 | MNIST                  | 60,000             | 10,000               | 28x28      | Downloadable through torchvision API   |
| 2 | CIFAR100               | 50,000             | 10,000               | 32x32      | Downloadable through torchvision API   |
| 3 | VWW                    | 40,775             | 8,059                | 224x224    | Based on COCO dataset                  |
| 4 | Imagenet10             | 385,244            | 15,011               | 224x224    | Subset of Imagenet2012 with 10 classes |
| 5 | Imagenet16             | 180,119            | 42,437               | 224x224    | Subset of Imagenet2012 with 16 classes |
| 6 | Imagenet               | 1,282,168          | 50,000               | 224x224    | Imagenet2012                           |
| 7 | VOC2007 (Detection)    | 5,011              | 4,952                | 500xH/Wx500| 20 classes, 24,640 annotated objects   |
| 8 | VOC2012 (Detection)    | 11,530 (train/val) | N/A                  | 500xH/Wx500| 20 classes, 27,450 annotated objects   |
| 9 | COCO2017 (Detection)   | 117,266, 5,000(val)| 40,670               | 300x300    | 80 Classes, 1.5M object instances      |


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
- The implementation of models on Imagenet dataset: [pytorch/vision](https://github.com/pytorch/vision)
