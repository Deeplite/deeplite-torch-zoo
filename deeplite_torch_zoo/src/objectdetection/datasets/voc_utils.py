import os
import argparse
import xml.etree.ElementTree as ET

from pathlib import Path
from tqdm import tqdm


def parse_voc_annotation(data_path, file_type, anno_path, use_difficult_bbox=False):
    class_names = []
    img_inds_file = os.path.join(data_path, "ImageSets", "Main", file_type + ".txt")
    with open(img_inds_file, "r") as f:
        lines = f.readlines()
        image_ids = [line.strip() for line in lines]

    with open(anno_path, "a") as f:
        for image_id in tqdm(image_ids):
            image_path = os.path.join(data_path, "JPEGImages", image_id + ".jpg")
            annotation = image_path
            label_path = os.path.join(data_path, "Annotations", image_id + ".xml")
            root = ET.parse(label_path).getroot()
            objects = root.findall("object")
            for obj in objects:
                difficult = obj.find("difficult").text.strip()
                if (not use_difficult_bbox) and (
                    int(difficult) == 1
                ):  # difficult
                    continue
                bbox = obj.find("bndbox")
                name = obj.find("name").text.lower().strip()
                if name not in class_names:
                    class_names.append(name)
                xmin = bbox.find("xmin").text.strip()
                ymin = bbox.find("ymin").text.strip()
                xmax = bbox.find("xmax").text.strip()
                ymax = bbox.find("ymax").text.strip()
                annotation += " " + ",".join([xmin, ymin, xmax, ymax, str(name)])
            annotation += "\n"
            f.write(annotation)
    return len(image_ids), class_names


def prepare_voc_data(train_data_paths, test_data_paths, data_root_annotation, train_test_split):
    print("Preparing VOC dataset for YOLO. Onetime process...")
    train_annotation_path = os.path.join(
        str(data_root_annotation), "train_annotation.txt"
    )
    if os.path.exists(train_annotation_path):
        os.remove(train_annotation_path)

    test_annotation_path = os.path.join(
        str(data_root_annotation), "test_annotation.txt"
    )
    if os.path.exists(test_annotation_path):
        os.remove(test_annotation_path)

    class_names_file_path = os.path.join(
        str(data_root_annotation), "class_names.txt"
    )
    if os.path.exists(class_names_file_path):
        os.remove(class_names_file_path)

    train_file_tag, test_file_tag = train_test_split

    len_train_total = 0
    class_names_train = []
    for train_data_path in train_data_paths:
        len_train, class_names = parse_voc_annotation(
            train_data_path,
            train_file_tag,
            train_annotation_path,
            use_difficult_bbox=False,
        )
        len_train_total += len_train
        class_names_train += class_names
    class_names_train = set(class_names_train)

    len_test_total = 0
    class_names_test = []
    for test_data_path in test_data_paths:
        len_test, class_names = parse_voc_annotation(
            test_data_path,
            test_file_tag,
            test_annotation_path,
            use_difficult_bbox=False,
        )
        len_test_total += len_test
        class_names_test += class_names
    class_names_test = set(class_names_test)

    with open(class_names_file_path, 'w') as f:
        f.write(' '.join(sorted(list(class_names_train))))

    print(f"The number of images for train and test are: \
            train : {len_train} | test : {len_test}. The number of classes is {len(class_names_train)}".format(len_train, len_test))


def prepare_yolo_voc_data(vockit_data_root, annotation_path, standard_voc_format=True, is_07_subset=False):

    Path(annotation_path).mkdir(parents=True, exist_ok=True)

    train_anno_path = os.path.join(str(annotation_path), "train_annotation.txt")
    test_anno_path = os.path.join(str(annotation_path), "test_annotation.txt")

    if standard_voc_format:
        train_data_paths = [os.path.join(vockit_data_root, "VOC2007"),
                os.path.join(vockit_data_root, "VOC2012")] if not is_07_subset \
                    else [os.path.join(vockit_data_root, "VOC2007"),]
        test_data_paths = [os.path.join(vockit_data_root, "VOC2007"),]
    else:
        train_data_paths, test_data_paths = [vockit_data_root,], [vockit_data_root,]

    train_test_split = ('trainval', 'test') if not is_07_subset else ('train', 'val')

    if not (os.path.exists(train_anno_path) and os.path.exists(test_anno_path)):
        prepare_voc_data(train_data_paths, test_data_paths, annotation_path, train_test_split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vockid_path",
        type=str,
        default="/neutrino/datasets/VOCdevkit/",
        help="VOCKIT path",
    )
    parser.add_argument(
        "--annotation_path", type=str, default="../data", help="Annotation data path"
    )
    args = parser.parse_args()

    prepare_voc_data(args.vockid_path, args.annotation_path)
