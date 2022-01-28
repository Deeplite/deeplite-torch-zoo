from deeplite_torch_zoo.wrappers.datasets import *
from deeplite_torch_zoo.wrappers.models import *
from deeplite_torch_zoo.utils.registry import MODEL_WRAPPER_REGISTRY


__all__ = ["get_data_splits_by_name", "get_model_by_name", "list_models"]


def normalize_model_name(net):
    if "yolo" in net:
        return "yolo"
    if "unet" in net:
        return "unet"
    if "ssd300" in net:
        return "ssd300"
    return net


def get_data_splits_by_name(data_root="", dataset_name="", model_name=None, **kwargs):
    """
    The datasets function calls in the format of (get_`dataset_name`_for_`model_name`).
    Except for classification since the datasets format for classification models is the same.
    The function calls for classification models are in the format (get_`dataset_name`)

    returns datasplits in the following format:
    {
       'train': train_data_loader,
       'test' : test_data_loader
    }
    """
    dataset_name = dataset_name.lower()
    func = f"get_{dataset_name}"
    if model_name is not None:
        model_name = normalize_model_name(model_name)
        model_name = model_name.lower()
        func = f"get_{dataset_name}_for_{model_name}"

    assert func in globals(), f"function {func} doesn't exist"
    return globals()[func](data_root=data_root, **kwargs)


def get_model_by_name(
    model_name="", dataset_name="", pretrained=False, progress=False, fp16=False, device="cuda"
):
    """
    The models function calls in the format of (`model_name`_`dataset_name`) all lower case.
    :param pretrained: Loads the pretrained model's wieghts.
    :param device: Loads the model either on a gpu (`cuda`, `cuda:device_id`) or cpu.

    returns a model (pretrained or fresh) with respect to the dataset
    """
    model_name = model_name.lower()
    model_func = MODEL_WRAPPER_REGISTRY.get((model_name, dataset_name))
    model = model_func(pretrained=pretrained, progress=progress, device=device)
    if fp16:
        model = model.half()

    return model


def list_models(key_word="*"):
    """
    A helper function to list all existing models or dataset calls
    It takes a `model_name` or a `dataset_name` and prints all matching function calls
    """
    matched_models = {}

    models_data = MODEL_WRAPPER_REGISTRY.registry_dict
    for model_key in models_data:
        model_name = model_key[0]
        dataset_name = model_key[1] or ''
        if key_word=="*" or key_word in model_name or key_word in dataset_name:
            matched_models[model_name] = matched_models.get(model_name,[])
            matched_models[model_name].append(model_key)

    print("_" * 80)
    print(f"{'Available Models':25} |Trained on datasets")
    print("-" * 80)
    for model_name, model_dataset_keys in matched_models.items():
        print(f"{model_name:{25}} | ", end='')

        matched_datasets = []
        for _, matched_dataset in model_dataset_keys:
            if matched_dataset:
                matched_datasets.append(matched_dataset)
        matched_datasets_str = ", ".join(matched_datasets)
        print(matched_datasets_str)
    print("_" * 80)


def get_models_names_for(dataset_name="imagenet"):
    dataset_name = f"_{dataset_name}*"
    model_names = []
    for _f in globals().keys():
        _f += "*"
        if dataset_name in _f:
            model_name = _f.replace(dataset_name, '')
            assert model_name not in model_names
            model_names.append(model_name)
    return model_names
