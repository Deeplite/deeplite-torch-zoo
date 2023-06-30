from deeplite_torch_zoo import get_eval_function, list_models
from deeplite_torch_zoo.api.eval import *

classification_eval_list = (classification_eval, )

objectdetection_eval_list = (
    yolo_eval_voc,
    yolo_eval_coco,
)


def test_eval_classification():
    all_classification_models = list_models(
        task_type_filter="classification", print_table=False
    )

    for (model_name, dataset_name) in all_classification_models:
        funct = get_eval_function(model_name=model_name, dataset_name=dataset_name)
        assert funct in classification_eval_list


def test_eval_objectdetection():
    all_od_models = list_models(
        task_type_filter="object_detection", print_table=False
    )
    for (model_name, dataset_name) in all_od_models:
        eval_fn = get_eval_function(model_name=model_name, dataset_name=dataset_name)
        assert eval_fn in objectdetection_eval_list
