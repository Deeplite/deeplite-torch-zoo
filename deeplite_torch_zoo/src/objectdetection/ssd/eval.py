import os

import numpy as np
import torch

from vision.utils import box_utils, measurements


def group_annotation_by_class(dataloader):
    true_case_stat = {}
    all_gt_boxes = {}
    all_difficult_cases = {}
    for i, _ in enumerate(dataloader.dataset):
        image_id, annotation = dataloader.dataset.get_annotation(i)
        gt_boxes, classes, is_difficult = annotation
        gt_boxes = torch.from_numpy(gt_boxes)
        for i, difficult in enumerate(is_difficult):
            class_index = int(classes[i])
            gt_box = gt_boxes[i]
            if not difficult:
                true_case_stat[class_index] = true_case_stat.get(class_index, 0) + 1

            if class_index not in all_gt_boxes:
                all_gt_boxes[class_index] = {}
            if image_id not in all_gt_boxes[class_index]:
                all_gt_boxes[class_index][image_id] = []
            all_gt_boxes[class_index][image_id].append(gt_box)
            if class_index not in all_difficult_cases:
                all_difficult_cases[class_index] = {}
            if image_id not in all_difficult_cases[class_index]:
                all_difficult_cases[class_index][image_id] = []
            all_difficult_cases[class_index][image_id].append(difficult)

    for class_index in all_gt_boxes:
        for image_id in all_gt_boxes[class_index]:
            all_gt_boxes[class_index][image_id] = torch.stack(
                all_gt_boxes[class_index][image_id]
            )
    for class_index in all_difficult_cases:
        for image_id in all_difficult_cases[class_index]:
            all_difficult_cases[class_index][image_id] = torch.tensor(
                all_difficult_cases[class_index][image_id]
            )
    return true_case_stat, all_gt_boxes, all_difficult_cases


def compute_average_precision_per_class(
    num_true_cases,
    gt_boxes,
    difficult_cases,
    prediction_file,
    iou_threshold,
    use_2007_metric,
):
    with open(str(prediction_file)) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            t = line.rstrip().split(" ")
            image_ids.append(t[0])
            scores.append(float(t[1]))
            box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
            box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()
        for i, image_id in enumerate(image_ids):
            box = boxes[i]
            if image_id not in gt_boxes:
                false_positive[i] = 1
                continue

            gt_box = gt_boxes[image_id]
            ious = box_utils.iou_of(box, gt_box)
            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > iou_threshold:
                if difficult_cases[image_id][max_arg] == 0:
                    if (image_id, max_arg) not in matched:
                        true_positive[i] = 1
                        matched.add((image_id, max_arg))
                    else:
                        false_positive[i] = 1
            else:
                false_positive[i] = 1

    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases
    if use_2007_metric:
        return measurements.compute_voc2007_average_precision(precision, recall)
    else:
        return measurements.compute_average_precision(precision, recall)


def ssd_eval(
    predictor,
    dataloader,
    data_path,
    class_names,
    iou_threshold=0.5,
    use_2007_metric=True,
):
    true_case_stat, all_gb_boxes, all_difficult_cases = group_annotation_by_class(
        dataloader
    )
    results = []
    for i, _ in enumerate(dataloader.dataset):
        image = dataloader.dataset.get_image(i)
        boxes, labels, probs = predictor.predict(image)
        # print("Prediction: {:4f} seconds.".format(timer.end("Predict")))
        indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
        results.append(
            torch.cat(
                [
                    indexes.reshape(-1, 1),
                    labels.reshape(-1, 1).float(),
                    probs.reshape(-1, 1),
                    boxes + 1.0,  # matlab's indexes start from 1
                ],
                dim=1,
            )
        )
    results = torch.cat(results)
    for class_index, class_name in enumerate(class_names):
        if class_index == 0:
            continue  # ignore background
        prediction_path = os.path.join(
            str(data_path), "det_test_{}.txt".format(class_name)
        )
        with open(str(prediction_path), "w") as f:
            sub = results[results[:, 1] == class_index, :]
            for i in range(sub.size(0)):
                prob_box = sub[i, 2:].numpy()
                image_id = dataloader.dataset.ids[int(sub[i, 0])]
                print(str(image_id) + " " + " ".join([str(v) for v in prob_box]), file=f)
    aps = []
    results = {}
    # print("\n\nAverage Precision Per-class:")
    for class_index, class_name in enumerate(class_names):
        if class_index == 0:
            continue
        prediction_path = os.path.join(
            str(data_path), "det_test_{}.txt".format(class_name)
        )
        ap = compute_average_precision_per_class(
            true_case_stat[class_index],
            all_gb_boxes[class_index],
            all_difficult_cases[class_index],
            prediction_path,
            iou_threshold,
            use_2007_metric,
        )
        aps.append(ap)
        # print("{}: {}".format(class_name, ap))
        results[class_name] = ap

    results["mAP"] = sum(aps) / len(aps)
    # print("\nAverage Precision Across All Classes:{}".format(sum(aps) / len(aps)))
    return results