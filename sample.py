import argparse

import deeplite_torch_zoo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model/dataset args
    parser.add_argument(
        "--dataset", metavar="DATASET", default="cifar100", help="dataset to use"
    )
    parser.add_argument(
        "-r", "--data_root", metavar="PATH", default="", help="dataset data root path"
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, metavar="N", default=128, help="mini-batch size"
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        metavar="N",
        default=4,
        help="number of data loading workers",
    )
    parser.add_argument(
        "--arch", "-a", metavar="ARCH", default="resnet18", help="model architecture"
    )

    args = parser.parse_args()

    data_splits = deeplite_torch_zoo.get_data_splits_by_name(
        dataset_name=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_torch_workers=args.workers,
    )

    reference_model = deeplite_torch_zoo.get_model_by_name(
        model_name=args.arch, dataset_name=args.dataset, pretrained=True, progress=True
    )
