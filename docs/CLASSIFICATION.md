# Classification

## Datasets

|Dataset Name    | Traininng Instances | Testing Instances| Resolution | Remarks |
|   ---          |        ---          |        ---       |    ---     |   ---   |
|  MNSIT         |         60000       |        10000     |     28*28  | Downloadable through torchvision API|
|  CIFAR100      |         50000       |        10000     |     32*32  | Downloadable through torchvision API|
|  VWW           |         40,775      |        8,059     |    224*224 | Based on COCO dataset |
|  TinyImageNet  |    1,00,000         |        10000     |     64*64  | Subset of Imagenet with 100 classes |
|  Imagenet10    |      385,244        |        15,011    |    224*224 | Subset of Imagenet2012 with 10 classes|
|  Imagenet16    |      180,119        |        42,437    |    224*224 | Subset of Imagenet2012 with 16 classes (1. plant 2. cat 3. dog 4. car 5. building 6. person 7. chair 8. pen 9. shoes 10. bag 11. hill 12. bed 13. wine 14. fish 15. boat 16. plane)|
|  Imagenet    |      1,282,168       |        50,000    |    224*224 | Imagenet2012 |

## Wrapper Functions

* One can get different data splits (train, and test splits) using the wrapper function get_data_splits_by_name,

```{.python}
    data_splits = get_data_splits_by_name(
        dataset_name="cifar100", model_name="resnet18", batch_size=128
    )
    train_split = data_splits['train']
    test_split = data_splits['test']
```

* To get the desired model architecture, we have a wrapper function get_model_by_name, which requires exact model and dataset name.

```{.python}
    model = get_model_by_name(
        model_name="resnet18",
        dataset_name="cifar100",
        pretrained=True, # or False, if pretrained weights are not required
        device="cpu", # or "gpu"
    )
```
* To get the evaluation function, we have a wrapper function get_eval_by_name, which requires exact model and dataset name.

```{.python}
    test_loader = data_splits["test"]
    eval_function = get_eval_function(dataset_name="voc", model_name="vgg16_ssd")
    APs = eval_function(model, test_loader)
```

## Training on Custom Dataset

### Basic Training Example
One can get an idea of complete working pipeline of the Deeplite Torch Zoo classification by looking at the [CIFAR training script](../training_scripts/classification/cifar/train_cifar.py).

### ImageNet Training Example

One can train a full fledged model by using the ImageNet training script [train.py](../training_scripts/classification/imagenet/train.py). The following command could be used to run this script,

```
    $ python train.py --dataset-name ZOO_DATASET_NAME --data_dir DATA_ROOT_PATH --model MODEL_NAME --num-classes NUM_CLASSES -b BATCH_SIZE --epochs NUM_EPOCHS --pretrained --pretraining-dataset PRETRAINING_DATASET
```

## Available pretrained classification models.

The zoo enables to load any ImageNet-pretrained model from the [timm repo](https://github.com/rwightman/pytorch-image-models) as well as any ImageNet model from torchvision. In case the model names overlap with timm, the corresponding timm model is loaded. Below is the list of available models that were trained on a classification dataset other than ImageNet:


|         Available models         |              Source datasets              |
|             -----                |                 -----                     |
| densenet121                      | cifar100, imagenet                        |
| googlenet                        | cifar100, imagenet                        |
| lenet5                           | mnist                                     |
| mixnet_l                         | imagenet                                  |
| mlp2                             | mnist                                     |
| mlp4                             | mnist                                     |
| mlp8                             | mnist                                     |
| mobileone s0,s1,s2,s3,s4         | imagenet                                  |
| mobilenet_v1                     | cifar100, vww                             |
| mobilenet_v2_0_35                | imagenet10                                |
| mobilenet_v2                     | cifar100, imagenet, tinyimagenet          |
| mobilenetv3_large                | vww                                       |
| mobilenetv3_small                | vww                                       |
| pre_act_resnet18                 | cifar100                                  |
| resnet18                         | cifar100, imagenet, imagenet10,           |
|                                  | imagenet16, tinyimagenet, vww             |
| resnet34                         | imagenet, tinyimagenet                    |
| resnet50                         | cifar100, imagenet, imagenet16,           |
|                                  | tinyimagenet, vww                         |
| resnext29_2x64d                  | cifar100                                  |
| shufflenet_v2_1_0                | cifar100                                  |
| vgg19                            | cifar100, imagenet, tinyimagenet          |
