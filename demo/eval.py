
import sys
import os
import torch
import argparse
from pathlib import Path

from deeplite_torch_zoo.src.objectdetection.yolov5.models.yolo_tflite_detect_stitced import StitchedYoloTflite
from deeplite_torch_zoo.src.objectdetection.eval.coco.coco_evaluator import yolo_eval_coco
from deeplite_torch_zoo.src.objectdetection.eval.voc.voc_evaluator import yolo_eval_voc




FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  #


def run(
    weights=ROOT / 'weights/opt_model_st_int8_224px_ch_3.tflite', # Path to tflite model withouot Detect layer
    source='/neutrino/datasets/st_voc/voc/',  # Path to dataset root
    imgsz=224,  # inference size
    dataset_type='voc' # Either 'voc' or coco formats are supported
):
    model = StitchedYoloTflite(weights, device=torch.device("cuda"))
    if dataset_type == 'voc':
        mAP = yolo_eval_voc(model, source, img_size=imgsz, subclasses=["person"])
    elif dataset_type == 'coco':
        mAP = yolo_eval_coco(
            model, source, img_size=imgsz,
            subsample_categories=["person"], device="cuda"
        )
    else:
        raise(f"Dataset {dataset_type} format is not supported!")
    print(mAP)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/opt_model_st_int8_224px_ch_3.tflite', help='model path or triton URL')
    parser.add_argument('--source', type=str, default='/neutrino/datasets/st_voc/voc/', help='path to dataset root')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=224, help='inference size')
    parser.add_argument('--dataset_type', type=str, default='voc', help='voc or coco')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
