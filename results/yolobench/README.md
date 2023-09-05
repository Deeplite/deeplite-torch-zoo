<a name="readme-top"></a>

<div align="center">

  <h2 align="center">YOLOBench: a latency-accuracy benchmark of YOLO-based object detectors on embedded platforms</h3>

  <a href="https://arxiv.org/abs/2307.13901" target="_blank"><img src="https://img.shields.io/badge/arXiv-2307.13901-b31b1b.svg" alt="arXiv"></a>
</div>

## Datasets

There are currently accuracy results for four datasets (VOC, WIDERFACE, SKU-110K, COCO) and latency results on 4 embedded platforms (ARM CPU, x86 CPU, Jetson Nano GPU and Khadas VIM3 NPU).

#### VOC (fine-tuned from COCO pre-training) - [`VOC_finetuned.csv`](VOC_finetuned.csv)

Standard [PASCAL VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC). Each model fine-tuned for 100 epochs from COCO pre-trained weights.

#### SKU-110K (fine-tuned from COCO pre-training) - [`SKU-110K_finetuned.csv`](SKU-110K_finetuned.csv)

[SKU-110K retail items dataset](https://github.com/eg4000/SKU110K_CVPR19) by Trax Retail. Each model fine-tuned for 100 epochs from COCO pre-trained weights.

#### WIDER FACE (fine-tuned from COCO pre-training) - [`WIDERFACE_finetuned.csv`](WIDERFACE_finetuned.csv)

[WIDER FACE dataset](http://shuoyang1213.me/WIDERFACE/) for face detection. Each model fine-tuned for 100 epochs from COCO pre-trained weights.

#### MSCOCO (trained from scratch for 300 epochs on 640x640 images) - [`COCO_pretrain_300epochs.csv`](COCO_pretrain_300epochs.csv)

MSCOCO dataset. Each model trained for 300 epochs from scratch on 640x640 images. Final (best) weights from these training jobs were used to initialize fine-tuning jobs on other datasets. Accuracy data provided is measured on the minival set, no fine-tuning using corresponding image resolutions is done, models trained on 640x640 are evaluated on multiple resolutions as is.

#### VOC (trained from scratch for 100 epochs) - [`VOC_scratch_100epochs.csv`](VOC_scratch_100epochs.csv)

Accuracy data for the whole YOLOBench search space on the VOC dataset trained from scratch (random initialization) for 100 epochs.


#### Latency - [`YOLO_latency.csv`](YOLO_latency.csv)

Latency data for the whole YOLOBench search space measured on 4 embedded platforms (Intel CPU, Raspi4 ARM CPU, Jetson Nano GPU and Khadas VIM3 NPU).
