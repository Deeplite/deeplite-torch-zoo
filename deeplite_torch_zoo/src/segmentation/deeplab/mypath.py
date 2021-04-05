class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == "pascal":
            return "data/VOC/"  # folder that contains VOCdevkit/.
        elif dataset == "sbd":
            return "data/benchmark_RELEASE/"  # folder that contains dataset/.
        elif dataset == "cityscapes":
            return "/path/to/datasets/cityscapes/"  # foler that contains leftImg8bit/
        elif dataset == "coco":
            return "/path/to/datasets/coco/"
        else:
            print("Dataset {} not available.".format(dataset))
            raise NotImplementedError
