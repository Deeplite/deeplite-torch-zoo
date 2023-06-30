# Object Detection

## Datasets

| name   | Dataset (dataset_name) | Training Instances | Test Instances       | Resolution | Comments |
| --  | ---------------------- | ------------------ | --------------       | ---------- | -------- |
|  voc | VOC2007 (Detection)    | 5,011              | 4,952                | 500xH/Wx500| 20 classes, 24,640 annotated objects   |
| voc  | VOC2012 (Detection)    | 11,530 (train/val) | N/A                  | 500xH/Wx500| 20 classes, 27,450 annotated objects   |
| coco | COCO2017 (Detection)   | 117,266, 5,000(val)| 40,670               | 300x300    | 80 Classes, 1.5M object instances      |
| person_detection | COCO Person (Detection)| 39283(train/val)   | 1648                 | 300x300    | 1 Class                                |

Note: We will get combined VOC2007 and VOC2012 dataset when we will use name 'voc' in the wrapper functions.

## Wrapper Functions

* One can get different data splits (train, and test splits) using the wrapper function get_dataloaders,

```{.python}
    data_splits = get_dataloaders(
        dataset_name="voc", model_name="yolo-v5s", batch_size=64
    )
    train_split = data_splits['train']
    test_split = data_splits['test']
```

* To get the desired model architecture, we have a wrapper function get_model, which requires exact model and dataset name.

```{.python}
    model = get_model(
        model_name="yolo-v5s",
        dataset_name="voc",
        pretrained=True, # or False, if pretrained weights are not required
    )
```
* To get the evaluation function, we have a wrapper function get_eval_by_name, which requires exact model and dataset name.

```{.python}
    test_loader = data_splits["test"]
    eval_function = get_eval_function(dataset_name="voc", model_name="yolo-v5s")
    APs = eval_function(model, test_loader)
```

List of models and corresponding datasets used to train it can be found [here](#complete-list-of-models-and-datasets).


## Creating a custom model based on existing architecture

One could create a model with a custom number of classes, while loading the pretrained weights from one of the pretrained models available in the zoo for the modules where weight reusage is possible. An example below creates a ``yolo5_6m`` model with 8 output classes, with weights loaded from a COCO checkpoint (except for the layers whise shape depends on the number of classes):

```{.python}
    from deeplite_torch_zoo import create_model

    model = create_model(
        model_name="yolo5_6m",
        pretraining_dataset="coco",
        num_classes=8,
        pretrained=True, # or False, if pretrained weights are not required
    )
```

## Training on Custom Dataset

It needs to be ensured that the data format should follow either of the formats present in the available datasets with proper names: voc, coco, coco_person. One can train any yolo model with resume functionalities using [train_yolo.py](../training_scripts/object_detection/train_yolo.py). Command to run this script,

```
    $ python train_yolo.py
            --dataset ZOO_DATASET_NAME
            --img-dir DATA_IMG_ROOT_PATH
            --net ZOO_MODEL_NAME
            --hp_config scratch                 # Options: 'scratch', 'finetune'
            --test_img_res 224
            --train_img_res 224
            --num-classes NUM_CLASSES
            --weight_path WEIGHT_PATH
            --batch-size BATCH_SIZE
            --epochs NUM_EPOCHS
            --device 0

```

Similarly for training any SSD model, one can use [train_ssd.py](../deeplite_torch_zoo/src/objectdetection/ssd/train_ssd.py) with proper arguments. Also, to access the pretrained weights, one can add --pretrained in the arguments along with the --pretraining_source_dataset which will again be selected from the available options.


## Complete List Of Models and Datasets

List of all models with their names and corresponding datasets used to train it.


| Available models | Source datasets                    |
|------------------|----------------------------------- |
| mb1_ssd          | voc                                |
| mb2_ssd          | coco, voc                          |
| mb2_ssd_lite     | voc                                |
| resnet18_ssd     | voc                                |
| resnet34_ssd     | voc                                |
| resnet50_ssd     | voc                                |
| vgg16_ssd        | voc, wider_face                    |
| yolo3            | voc                                |
| yolo4l_leaky     | voc                                |
| yolo4l           | voc                                |
| yolo4m           | coco, voc                          |
| yolo4s           | coco, voc                          |
| yolo4x           | voc                                |
| yolo5_6l         | voc                                |
| yolo5_6m         | coco, voc                          |
| yolo5_6m_relu    | person_detection, voc              |
| yolo5_6ma        | coco                               |
| yolo5_6n         | coco, person_detection, voc, voc07 |
| yolo5_6n_hswish  | coco, voc                          |
| yolo5_6n_relu    | coco, person_detection, voc        |
| yolo5_6s         | coco, person_detection, voc, voc07 |
| yolo5_6s_hswish  | coco, voc                          |
| yolo5_6s_relu    | person_detection, voc              |
| yolo5_6sa        | coco, person_detection             |
| yolo5_6x         | voc                                |
