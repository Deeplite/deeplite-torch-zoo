# Installation

## Install from source (development version)

```
    $ git clone https://github.com/Deeplite/deeplite-torch-zoo.git
    $ git checkout tflite-stitched
    $ pip install .
    $ pip install -r requirements-demo.txt
```

## Install in dev mode

```
    $ git clone https://github.com/Deeplite/deeplite-torch-zoo.git
    $ pip install -e .
    $ pip install -r requirements-demo.txt

```


# Usage

## Evaluation

The models can be evaluated on either `coco` format or `voc` format.
`python demo/eval.py --source /neutrino/datasets/coco/ --dataset_type coco --weights path/to/tflite/model.tflite --imgsz xxx`


## Visualization and detections generation
`python demo/annotate.py --weights path/to/tflite/model.tflite --imgsz xxx --conf-thres 0.2 --line-width 3`

The output of the script can be found in `results/exp/labels` Visualization can be found in `results/epx`

In `labels` folder, a text file for each image containing person detections. `class_id, conf, cx, cy, w, h` all coordinates are normalized