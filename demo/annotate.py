

import argparse
import os
import platform
import sys
from pathlib import Path
import cv2
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from deeplite_torch_zoo.src.objectdetection.yolov5.models.yolo_tflite_detect_stitced import StitchedYoloTflite
from demo.image_loader import LoadImages
from demo.utils import xywh2xyxy, xyxy2xywh, Profile, scale_boxes, Annotator
from deeplite_torch_zoo.src.objectdetection.yolov5.utils.general import nms


def run(
        weights=ROOT / 'weights/opt_model_st_int8_224px_ch_3.tflite', # Path to tflite model withouot Detect layer
        source='/neutrino/datasets/st_voc/voc/',  # Path to dataset root
        imgsz=224,  # inference size
        dataset_type='voc', # Either 'voc' or coco formats are supported
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        project=ROOT / 'results',  # save results to project/name
        name=Path('exp'),
        line_width=10,
):
    source = str(source) 
    # Directories
    save_dir = Path(project) / name
    (save_dir / 'labels').mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    model = StitchedYoloTflite(weights, device=torch.device('cuda'))

    # Dataloader
    bs = 1  # batch_size

    dataset = LoadImages(source, img_size=imgsz)

    # Run inference
    #model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = model(im)[0]

        # NMS
        with dt[2]:
            pred = nms(xywh2xyxy(pred.cpu().numpy().squeeze(None)), conf_thres, iou_thres)


        # Process predictions
        for i, det in enumerate([pred]):  # per image
            seen += 1
            p, im0 = path, im0s.copy()

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0, line_width=line_width)
            with open(f'{txt_path}.txt', 'w') as f:
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (0, conf, *xywh)
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        label = f'{conf:.2f}'
                        annotator.box_label(xyxy, label)
            im0 = annotator.result()
            cv2.imwrite(save_path, im0)
        # Print time (inference-only)
        print(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, imgsz)}' % t)
    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}"
    print(f"Results saved to {save_dir}{s}")
 

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/opt_model_st_int8_224px_ch_3.tflite', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'images', help='path to folder of images')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=224, help='inference size')
    parser.add_argument('--line-width', type=int, default=10, help='bounding box thickness')
    parser.add_argument('--conf-thres', type=float, default=0.01, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
