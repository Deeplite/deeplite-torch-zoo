# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import os
import pickle
import xml.etree.ElementTree as ET

import numpy as np
from mean_average_precision import MetricBuilder


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        obj_struct["pose"] = obj.find("pose").text
        obj_struct["truncated"] = int(obj.find("truncated").text)
        obj_struct["difficult"] = int(obj.find("difficult").text)
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(float(bbox.find("xmin").text)),
            int(float(bbox.find("ymin").text)),
            int(float(bbox.find("xmax").text)),
            int(float(bbox.find("ymax").text)),
        ]
        objects.append(obj_struct)

    return objects


def voc_eval(
    detpath,
    annopath,
    imagesetfile,
    classname,
    cachedir,
    ovthresh=0.5,
    use_07_metric=False,
):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, "annots.pkl")
    # read list of images
    with open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            # if i % 100 == 0:
            #     print ('Reading annotation for {:d}/{:d}'.format(
            #         i + 1, len(imagenames)))
        # save
        # print ('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, "wb") as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, "rb") as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == classname and
            not obj['difficult']]
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    # read dets
    detfile = detpath.format(classname)
    if not os.path.isfile(detfile):
        return 0, 0, 0

    with open(detfile, "r") as f:
        lines = f.readlines()

    splitlines = [x.strip().split(" ") for x in lines]

    predictions = {}
    for x in splitlines:
        if x[0] not in predictions:
                predictions[x[0]] = {'bbox': [[int(z) for z in x[2:]]], 'conf': [float(x[1])]}
        else:
            predictions[x[0]]['bbox'].append([int(z) for z in x[2:]])
            predictions[x[0]]['conf'].append(float(x[1]))

    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)

    for imagename in imagenames:
        gt = []
        pred = []
        for box_num, bbox in enumerate(class_recs[imagename]['bbox']):
            gt.append(np.array((*bbox, 0, int(class_recs[imagename]['difficult'][box_num]), 0)))
        if imagename in predictions:
            for idx, bbox in enumerate(predictions[imagename]['bbox']):
                pred.append(np.array([*bbox, 0, predictions[imagename]['conf'][idx]]))
        metric_fn.add(np.array(pred), np.array(gt))

    if not use_07_metric:
        return metric_fn.value(iou_thresholds=ovthresh)['mAP']
    else:
        return metric_fn.value(iou_thresholds=ovthresh,
            recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']
