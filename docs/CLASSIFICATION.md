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
        progress=False, # or True, if a progressbar is required
        device="cpu", # or "gpu"
    )
```
* To get the evaluation function, we have a wrapper function get_eval_by_name, which requires exact model and dataset name.

```{.python}
    test_loader = data_splits["test"]
    eval_function = get_eval_function(dataset_name="voc", model_name="vgg16_ssd")
    APs = eval_function(model, test_loader)
```

List of models and corresponding datasets used to train it can be found [here](#complete-list-of-models-and-datasets).

## Training on Custom Dataset

### Basic Training Example
One can get an idea of complete working pipeline of the deeplite_torch_zoo classification using [train_classifier.py](../examples/train_classifier.py). It needs to be ensured that the data format should follow either of the formats present in the available datasets.

```
    $ python train_classifier.py --dataset DATA_FORMAT --data_root DATA_ROOT_PATH -a MODEL_ARCHITECTURE
```

### Advanced Training Example
One can train a full fledged model with resume functionalities using [train.py](../training_scripts/classification/train.py). Command to run this script, 

```
    $ python train.py --dataset-name DATA_FORMAT --data_dir DATA_ROOT_PATH --model MODEL_ARCHITECTURE --num-classes NUM_CLASSES --img-size 3 224 224 -b BATCH_SIZE --epochs NUM_EPOCHS 
```

## Complete List Of Models and Datasets

List of all models with their names and corresponding datasets used to train it.

|         Available models         |              Source datasets              |
|             -----                |                 -----                     |
| adv_inception_v3                 | imagenet                                  |
| alexnet                          | imagenet                                  |
| convnext_base_384_in22ft1k       | imagenet                                  |
| convnext_base                    | imagenet                                  |
| convnext_base_in22ft1k           | imagenet                                  |
| convnext_base_in22k              | imagenet                                  |
| convnext_large_384_in22ft1k      | imagenet                                  |
| convnext_large                   | imagenet                                  |
| convnext_large_in22ft1k          | imagenet                                  |
| convnext_large_in22k             | imagenet                                  |
| convnext_small                   | imagenet                                  |
| convnext_tiny_hnf                | imagenet                                  |
| convnext_tiny                    | imagenet                                  |
| convnext_xlarge_384_in22ft1k     | imagenet                                  |
| convnext_xlarge_in22ft1k         | imagenet                                  |
| convnext_xlarge_in22k            | imagenet                                  |
| cspdarknet53                     | imagenet                                  |
| cspresnet50                      | imagenet                                  |
| cspresnet50d                     | imagenet                                  |
| cspresnet50w                     | imagenet                                  |
| cspresnext50                     | imagenet                                  |
| darknet53                        | imagenet                                  |
| densenet121                      | cifar100, imagenet                        |
| densenet121d                     | imagenet                                  |
| densenet161                      | imagenet                                  |
| densenet169                      | imagenet                                  |
| densenet201                      | imagenet                                  |
| densenet264                      | imagenet                                  |
| densenet264d_iabn                | imagenet                                  |
| densenetblur121d                 | imagenet                                  |
| dla102                           | imagenet                                  |
| dla102x2                         | imagenet                                  |
| dla102x                          | imagenet                                  |
| dla169                           | imagenet                                  |
| dla34                            | imagenet                                  |
| dla46_c                          | imagenet                                  |
| dla46x_c                         | imagenet                                  |
| dla60                            | imagenet                                  |
| dla60_res2net                    | imagenet                                  |
| dla60_res2next                   | imagenet                                  |
| dla60x_c                         | imagenet                                  |
| dla60x                           | imagenet                                  |
| dpn107                           | imagenet                                  |
| dpn131                           | imagenet                                  |
| dpn68                            | imagenet                                  |
| dpn68b                           | imagenet                                  |
| dpn92                            | imagenet                                  |
| dpn98                            | imagenet                                  |
| efficientnet_b0                  | imagenet                                  | 
| efficientnet_b1                  | imagenet                                  | 
| efficientnet_b1_pruned           | imagenet                                  | 
| efficientnet_b2                  | imagenet                                  | 
| efficientnet_b2_pruned           | imagenet                                  | 
| efficientnet_b2a                 | imagenet                                  | 
| efficientnet_b3                  | imagenet                                  | 
| efficientnet_b3_pruned           | imagenet                                  | 
| efficientnet_b3a                 | imagenet                                  | 
| efficientnet_b4                  | imagenet                                  | 
| efficientnet_b5                  | imagenet                                  | 
| efficientnet_b6                  | imagenet                                  | 
| efficientnet_b7                  | imagenet                                  | 
| efficientnet_b8                  | imagenet                                  | 
| efficientnet_cc_b0_4e            | imagenet                                  | 
| efficientnet_cc_b0_8e            | imagenet                                  | 
| efficientnet_cc_b1_8e            | imagenet                                  | 
| efficientnet_el                  | imagenet                                  | 
| efficientnet_el_pruned           | imagenet                                  | 
| efficientnet_em                  | imagenet                                  | 
| efficientnet_es                  | imagenet                                  | 
| efficientnet_es_pruned           | imagenet                                  | 
| efficientnet_l2                  | imagenet                                  | 
| efficientnet_lite0               | imagenet                                  | 
| efficientnet_lite1               | imagenet                                  | 
| efficientnet_lite2               | imagenet                                  | 
| efficientnet_lite3               | imagenet                                  | 
| efficientnet_lite4               | imagenet                                  | 
| efficientnetv2_l                 | imagenet                                  | 
| efficientnetv2_m                 | imagenet                                  | 
| efficientnetv2_rw_m              | imagenet                                  | 
| efficientnetv2_rw_s              | imagenet                                  | 
| efficientnetv2_rw_t              | imagenet                                  | 
| efficientnetv2_s                 | imagenet                                  | 
| efficientnetv2_xl                | imagenet                                  | 
| ens_adv_inception_resnet_v2      | imagenet                                  | 
| ese_vovnet19b_dw                 | imagenet                                  | 
| ese_vovnet19b_slim_dw            | imagenet                                  | 
| ese_vovnet19b_slim               | imagenet                                  | 
| ese_vovnet39b_evos               | imagenet                                  | 
| ese_vovnet39b                    | imagenet                                  | 
| ese_vovnet57b                    | imagenet                                  | 
| ese_vovnet99b_iabn               | imagenet                                  | 
| ese_vovnet99b                    | imagenet                                  | 
| fbnetc_100                       | imagenet                                  | 
| fbnetv3_b                        | imagenet                                  | 
| fbnetv3_d                        | imagenet                                  | 
| fbnetv3_g                        | imagenet                                  | 
| gc_efficientnetv2_rw_t           | imagenet                                  | 
| gcresnet33ts                     | imagenet                                  | 
| gcresnet50t                      | imagenet                                  | 
| gcresnext26ts                    | imagenet                                  | 
| gcresnext50ts                    | imagenet                                  | 
| gernet_l                         | imagenet                                  | 
| gernet_m                         | imagenet                                  | 
| gernet_s                         | imagenet                                  | 
| ghostnet_050                     | imagenet                                  | 
| ghostnet_100                     | imagenet                                  | 
| ghostnet_130                     | imagenet                                  | 
| gluon_inception_v3               | imagenet                                  | 
| gluon_resnet101_v1b              | imagenet                                  | 
| gluon_resnet101_v1c              | imagenet                                  | 
| gluon_resnet101_v1d              | imagenet                                  | 
| gluon_resnet101_v1s              | imagenet                                  | 
| gluon_resnet152_v1b              | imagenet                                  | 
| gluon_resnet152_v1c              | imagenet                                  | 
| gluon_resnet152_v1d              | imagenet                                  | 
| gluon_resnet152_v1s              | imagenet                                  | 
| gluon_resnet18_v1b               | imagenet                                  | 
| gluon_resnet34_v1b               | imagenet                                  | 
| gluon_resnet50_v1b               | imagenet                                  | 
| gluon_resnet50_v1c               | imagenet                                  | 
| gluon_resnet50_v1d               | imagenet                                  | 
| gluon_resnet50_v1s               | imagenet                                  | 
| gluon_resnext101_32x4d           | imagenet                                  | 
| gluon_resnext101_64x4d           | imagenet                                  | 
| gluon_resnext50_32x4d            | imagenet                                  | 
| gluon_senet154                   | imagenet                                  | 
| gluon_seresnext101_32x4d         | imagenet                                  | 
| gluon_seresnext101_64x4d         | imagenet                                  | 
| gluon_seresnext50_32x4d          | imagenet                                  | 
| gluon_xception65                 | imagenet                                  | 
| googlenet                        | cifar100, imagenet                        | 
| halo2botnet50ts_256              | imagenet                                  | 
| halonet26t                       | imagenet                                  |
| halonet50ts                      | imagenet                                  |
| halonet_h1                       | imagenet                                  |
| hrnet_w18                        | imagenet                                  |
| hrnet_w18_small                  | imagenet                                  |
| hrnet_w18_small_v2               | imagenet                                  |
| hrnet_w30                        | imagenet                                  |
| hrnet_w32                        | imagenet                                  |
| hrnet_w40                        | imagenet                                  |
| hrnet_w44                        | imagenet                                  |
| hrnet_w48                        | imagenet                                  |
| hrnet_w64                        | imagenet                                  |
| inception_resnet_v2              | imagenet                                  |
| inception_v3                     | imagenet                                  |
| inception_v4                     | imagenet                                  |
| jx_nest_base                     | imagenet                                  |
| jx_nest_small                    | imagenet                                  |
| jx_nest_tiny                     | imagenet                                  | 
| lcnet_035                        | imagenet                                  | 
| lcnet_050                        | imagenet                                  | 
| lcnet_075                        | imagenet                                  | 
| lcnet_100                        | imagenet                                  | 
| lcnet_150                        | imagenet                                  | 
| legacy_senet154                  | imagenet                                  | 
| legacy_seresnet101               | imagenet                                  | 
| legacy_seresnet152               | imagenet                                  | 
| legacy_seresnet18                | imagenet                                  | 
| legacy_seresnet34                | imagenet                                  | 
| legacy_seresnet50                | imagenet                                  | 
| legacy_seresnext101_32x4d        | imagenet                                  | 
| legacy_seresnext26_32x4d         | imagenet                                  | 
| legacy_seresnext50_32x4d         | imagenet                                  | 
| lenet5                           | mnist                                     | 
| mixnet_l                         | imagenet                                  | 
| mixnet_m                         | imagenet                                  | 
| mixnet_s                         | imagenet                                  | 
| mixnet_xl                        | imagenet                                  | 
| mixnet_xxl                       | imagenet                                  | 
| mlp2                             | mnist                                     | 
| mlp4                             | mnist                                     | 
| mlp8                             | mnist                                     | 
| mnasnet0_5                       | imagenet                                  | 
| mnasnet0_75                      | imagenet                                  | 
| mnasnet1_0                       | imagenet                                  | 
| mnasnet1_3                       | imagenet                                  | 
| mnasnet_050                      | imagenet                                  | 
| mnasnet_075                      | imagenet                                  | 
| mnasnet_100                      | imagenet                                  | 
| mnasnet_140                      | imagenet                                  | 
| mnasnet_a1                       | imagenet                                  | 
| mnasnet_b1                       | imagenet                                  | 
| mnasnet_small                    | imagenet                                  | 
| mobilenet_v1                     | cifar100, vww                             | 
| mobilenet_v2_0_35                | imagenet10                                | 
| mobilenet_v2                     | cifar100, imagenet, tinyimagenet          | 
| mobilenet_v3_large               | imagenet                                  | 
| mobilenet_v3_small               | imagenet                                  | 
| mobilenetv2_035                  | imagenet                                  | 
| mobilenetv2_050                  | imagenet                                  | 
| mobilenetv2_075                  | imagenet                                  | 
| mobilenetv2_100                  | imagenet                                  | 
| mobilenetv2_110d                 | imagenet                                  | 
| mobilenetv2_120d                 | imagenet                                  | 
| mobilenetv2_140                  | imagenet                                  | 
| mobilenetv3_large                | vww                                       | 
| mobilenetv3_small                | vww                                       | 
| nasnetalarge                     | imagenet                                  | 
| nest_base                        | imagenet                                  | 
| nest_small                       | imagenet                                  | 
| nest_tiny                        | imagenet                                  | 
| nf_ecaresnet101                  | imagenet                                  | 
| nf_ecaresnet26                   | imagenet                                  | 
| nf_ecaresnet50                   | imagenet                                  | 
| nf_regnet_b0                     | imagenet                                  | 
| nf_regnet_b1                     | imagenet                                  | 
| nf_regnet_b2                     | imagenet                                  | 
| nf_regnet_b3                     | imagenet                                  | 
| nf_regnet_b4                     | imagenet                                  | 
| nf_regnet_b5                     | imagenet                                  | 
| nf_resnet101                     | imagenet                                  | 
| nf_resnet26                      | imagenet                                  | 
| nf_resnet50                      | imagenet                                  | 
| nf_seresnet101                   | imagenet                                  | 
| nf_seresnet26                    | imagenet                                  | 
| nf_seresnet50                    | imagenet                                  | 
| pnasnet5large                    | imagenet                                  | 
| pre_act_resnet18                 | cifar100                                  | 
| q_googlenet                      | imagenet                                  | 
| q_inception_v3                   | imagenet                                  | 
| q_mobilenet_v2                   | imagenet                                  | 
| q_mobilenet_v3_large             | imagenet                                  | 
| q_resnet18                       | imagenet                                  | 
| q_resnet50                       | imagenet                                  | 
| q_resnext101_32x8d               | imagenet                                  | 
| q_shufflenet_v2_x0_5             | imagenet                                  | 
| q_shufflenet_v2_x1_0             | imagenet                                  | 
| q_shufflenet_v2_x1_5             | imagenet                                  | 
| q_shufflenet_v2_x2_0             | imagenet                                  | 
| regnetx_002                      | imagenet                                  |
| regnetx_004                      | imagenet                                  |
| regnetx_006                      | imagenet                                  |
| regnetx_008                      | imagenet                                  |
| regnetx_016                      | imagenet                                  |
| regnetx_032                      | imagenet                                  |
| regnetx_040                      | imagenet                                  |
| regnetx_064                      | imagenet                                  |
| regnetx_080                      | imagenet                                  |
| regnetx_120                      | imagenet                                  |
| regnetx_160                      | imagenet                                  |
| regnetx_320                      | imagenet                                  |
| regnety_002                      | imagenet                                  |
| regnety_004                      | imagenet                                  |
| regnety_006                      | imagenet                                  |
| regnety_008                      | imagenet                                  |
| regnety_016                      | imagenet                                  |
| regnety_032                      | imagenet                                  |
| regnety_040                      | imagenet                                  |
| regnety_064                      | imagenet                                  |
| regnety_080                      | imagenet                                  |
| regnety_120                      | imagenet                                  |
| regnety_160                      | imagenet                                  |
| regnety_320                      | imagenet                                  |
| regnetz_b16                      | imagenet                                  |
| regnetz_c16                      | imagenet                                  |
| regnetz_d32                      | imagenet                                  |
| regnetz_d8_evob                  | imagenet                                  |
| regnetz_d8_evos                  | imagenet                                  |
| regnetz_d8                       | imagenet                                  |
| regnetz_e8                       | imagenet                                  |
| repvgg_a2                        | imagenet                                  |
| repvgg_b0                        | imagenet                                  |
| repvgg_b1                        | imagenet                                  |
| repvgg_b1g4                      | imagenet                                  |
| repvgg_b2                        | imagenet                                  |
| repvgg_b2g4                      | imagenet                                  |
| repvgg_b3                        | imagenet                                  |
| repvgg_b3g4                      | imagenet                                  |
| res2net101_26w_4s                | imagenet                                  |
| res2net50_14w_8s                 | imagenet                                  |
| res2net50_26w_4s                 | imagenet                                  |
| res2net50_26w_6s                 | imagenet                                  |
| res2net50_26w_8s                 | imagenet                                  | 
| res2net50_48w_2s                 | imagenet                                  | 
| res2next50                       | imagenet                                  | 
| resnest101e                      | imagenet                                  | 
| resnest14d                       | imagenet                                  | 
| resnest200e                      | imagenet                                  | 
| resnest269e                      | imagenet                                  | 
| resnest26d                       | imagenet                                  | 
| resnest50d_1s4x24d               | imagenet                                  | 
| resnest50d_4s2x40d               | imagenet                                  | 
| resnest50d                       | imagenet                                  | 
| resnet101                        | imagenet                                  | 
| resnet101d                       | imagenet                                  | 
| resnet152                        | imagenet                                  | 
| resnet152d                       | imagenet                                  | 
| resnet18                         | cifar100, imagenet, imagenet10,           |
|                                  | imagenet16, tinyimagenet, vww             | 
| resnet18d                        | imagenet                                  | 
| resnet200                        | imagenet                                  | 
| resnet200d                       | imagenet                                  | 
| resnet26                         | imagenet                                  | 
| resnet26d                        | imagenet                                  | 
| resnet26t                        | imagenet                                  | 
| resnet32ts                       | imagenet                                  | 
| resnet33ts                       | imagenet                                  | 
| resnet34                         | imagenet, tinyimagenet                    | 
| resnet34d                        | imagenet                                  | 
| resnet50                         | cifar100, imagenet, imagenet16,           |
|                                  | tinyimagenet, vww                         | 
| resnet50_gn                      | imagenet                                  | 
| resnet50d                        | imagenet                                  | 
| resnet50t                        | imagenet                                  | 
| resnet51q                        | imagenet                                  | 
| resnet61q                        | imagenet                                  | 
| resnetblur18                     | imagenet                                  | 
| resnetblur50                     | imagenet                                  | 
| resnetrs101                      | imagenet                                  | 
| resnetrs152                      | imagenet                                  | 
| resnetrs200                      | imagenet                                  | 
| resnetrs270                      | imagenet                                  | 
| resnetrs350                      | imagenet                                  | 
| resnetrs420                      | imagenet                                  | 
| resnetrs50                       | imagenet                                  | 
| resnext101_32x4d                 | imagenet                                  | 
| resnext101_32x8d                 | imagenet                                  | 
| resnext101_64x4d                 | imagenet                                  | 
| resnext26ts                      | imagenet                                  | 
| resnext29_2x64d                  | cifar100                                  | 
| resnext50_32x4d                  | imagenet                                  | 
| resnext50d_32x4d                 | imagenet                                  | 
| rexnet_100                       | imagenet                                  | 
| rexnet_130                       | imagenet                                  | 
| rexnet_150                       | imagenet                                  |
| rexnet_200                       | imagenet                                  |
| rexnetr_100                      | imagenet                                  |
| rexnetr_130                      | imagenet                                  |
| rexnetr_150                      | imagenet                                  |
| rexnetr_200                      | imagenet                                  |
| selecsls42                       | imagenet                                  |
| selecsls42b                      | imagenet                                  |
| selecsls60                       | imagenet                                  |
| selecsls60b                      | imagenet                                  |
| selecsls84                       | imagenet                                  |
| semnasnet_050                    | imagenet                                  |
| semnasnet_075                    | imagenet                                  |
| semnasnet_100                    | imagenet                                  |
| semnasnet_140                    | imagenet                                  |
| shufflenet_v2_1_0                | cifar100                                  |
| shufflenet_v2_x0_5               | imagenet                                  |
| shufflenet_v2_x1_0               | imagenet                                  |
| shufflenet_v2_x1_5               | imagenet                                  |
| shufflenet_v2_x2_0               | imagenet                                  |
| skresnet18                       | imagenet                                  |
| skresnet34                       | imagenet                                  |
| skresnet50                       | imagenet                                  |
| skresnet50d                      | imagenet                                  |
| skresnext50_32x4d                | imagenet                                  |
| spnasnet_100                     | imagenet                                  |
| squeezenet1_0                    | imagenet                                  |
| squeezenet1_1                    | imagenet                                  |
| ssl_resnet18                     | imagenet                                  |
| ssl_resnet50                     | imagenet                                  |
| ssl_resnext101_32x16d            | imagenet                                  |
| ssl_resnext101_32x4d             | imagenet                                  |
| ssl_resnext101_32x8d             | imagenet                                  |
| ssl_resnext50_32x4d              | imagenet                                  |
| swsl_resnet18                    | imagenet                                  |
| swsl_resnet50                    | imagenet                                  |
| swsl_resnext101_32x16d           | imagenet                                  |
| swsl_resnext101_32x4d            | imagenet                                  |
| swsl_resnext101_32x8d            | imagenet                                  |
| swsl_resnext50_32x4d             | imagenet                                  |
| tf_efficientnet_b0_ap            | imagenet                                  |
| tf_efficientnet_b0               | imagenet                                  |
| tf_efficientnet_b0_ns            | imagenet                                  |
| tf_efficientnet_b1_ap            | imagenet                                  |
| tf_efficientnet_b1               | imagenet                                  |
| tf_efficientnet_b1_ns            | imagenet                                  |
| tf_efficientnet_b2_ap            | imagenet                                  |
| tf_efficientnet_b2               | imagenet                                  |
| tf_efficientnet_b2_ns            | imagenet                                  |
| tf_efficientnet_b3_ap            | imagenet                                  |
| tf_efficientnet_b3               | imagenet                                  |
| tf_efficientnet_b3_ns            | imagenet                                  |
| tf_efficientnet_b4_ap            | imagenet                                  |
| tf_efficientnet_b4               | imagenet                                  |
| tf_efficientnet_b4_ns            | imagenet                                  |
| tf_efficientnet_b5_ap            | imagenet                                  |
| tf_efficientnet_b5               | imagenet                                  |
| tf_efficientnet_b5_ns            | imagenet                                  |
| tf_efficientnet_b6_ap            | imagenet                                  |
| tf_efficientnet_b6               | imagenet                                  |
| tf_efficientnet_b6_ns            | imagenet                                  |
| tf_efficientnet_b7_ap            | imagenet                                  |
| tf_efficientnet_b7               | imagenet                                  |
| tf_efficientnet_b7_ns            | imagenet                                  |
| tf_efficientnet_b8_ap            | imagenet                                  |
| tf_efficientnet_b8               | imagenet                                  |
| tf_efficientnet_cc_b0_4e         | imagenet                                  |
| tf_efficientnet_cc_b0_8e         | imagenet                                  |
| tf_efficientnet_cc_b1_8e         | imagenet                                  |
| tf_efficientnet_el               | imagenet                                  |
| tf_efficientnet_em               | imagenet                                  |
| tf_efficientnet_es               | imagenet                                  |
| tf_efficientnet_l2_ns_475        | imagenet                                  |
| tf_efficientnet_l2_ns            | imagenet                                  |
| tf_efficientnet_lite0            | imagenet                                  |
| tf_efficientnet_lite1            | imagenet                                  |
| tf_efficientnet_lite2            | imagenet                                  |
| tf_efficientnet_lite3            | imagenet                                  |
| tf_efficientnet_lite4            | imagenet                                  |
| tf_efficientnetv2_b0             | imagenet                                  |
| tf_efficientnetv2_b1             | imagenet                                  |
| tf_efficientnetv2_b2             | imagenet                                  |
| tf_efficientnetv2_b3             | imagenet                                  |
| tf_efficientnetv2_l              | imagenet                                  |
| tf_efficientnetv2_l_in21ft1k     | imagenet                                  |
| tf_efficientnetv2_l_in21k        | imagenet                                  |
| tf_efficientnetv2_m              | imagenet                                  |
| tf_efficientnetv2_m_in21ft1k     | imagenet                                  |
| tf_efficientnetv2_m_in21k        | imagenet                                  |
| tf_efficientnetv2_s              | imagenet                                  |
| tf_efficientnetv2_s_in21ft1k     | imagenet                                  |
| tf_efficientnetv2_s_in21k        | imagenet                                  |
| tf_efficientnetv2_xl_in21ft1k    | imagenet                                  |
| tf_efficientnetv2_xl_in21k       | imagenet                                  |
| tf_inception_v3                  | imagenet                                  |
| tf_mixnet_l                      | imagenet                                  |
| tf_mixnet_m                      | imagenet                                  |
| tf_mixnet_s                      | imagenet                                  |
| tf_mobilenetv3_large_075         | imagenet                                  |
| tf_mobilenetv3_large_100         | imagenet                                  |
| tf_mobilenetv3_large_minimal_100 | imagenet                                  |
| tf_mobilenetv3_small_075         | imagenet                                  |
| tf_mobilenetv3_small_100         | imagenet                                  |
| tf_mobilenetv3_small_minimal_100 | imagenet                                  |
| tinynet_a                        | imagenet                                  |
| tinynet_b                        | imagenet                                  |
| tinynet_c                        | imagenet                                  |
| tinynet_d                        | imagenet                                  |
| tinynet_e                        | imagenet                                  |
| tnt_b_patch16_224                | imagenet                                  |
| tnt_s_patch16_224                | imagenet                                  |
| tresnet_l_448                    | imagenet                                  |
| tresnet_l                        | imagenet                                  |
| tresnet_m_448                    | imagenet                                  |
| tresnet_m                        | imagenet                                  |
| tresnet_m_miil_in21k             | imagenet                                  |
| tresnet_xl_448                   | imagenet                                  |
| tresnet_xl                       | imagenet                                  |
| vgg11_bn                         | imagenet                                  |
| vgg11                            | imagenet                                  |
| vgg13_bn                         | imagenet                                  |
| vgg13                            | imagenet                                  |
| vgg16_bn                         | imagenet                                  |
| vgg16                            | imagenet                                  |
| vgg19_bn                         | imagenet                                  |
| vgg19                            | cifar100, imagenet, tinyimagenet          |
| vovnet39a                        | imagenet                                  |
| vovnet57a                        | imagenet                                  |
| wide_resnet101_2                 | imagenet                                  |
| wide_resnet50_2                  | imagenet                                  |
| xception41                       | imagenet                                  |
| xception65                       | imagenet                                  |
| xception71                       | imagenet                                  |
| xception                         | imagenet                                  |
