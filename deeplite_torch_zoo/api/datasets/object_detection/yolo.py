# Ultralytics YOLO 🚀, AGPL-3.0 license

from deeplite_torch_zoo.utils import RANK

from deeplite_torch_zoo.src.object_detection.datasets.dataloader import get_dataloader
from ultralytics.yolo.data.utils import check_det_dataset
from ultralytics.yolo.utils import DEFAULT_CFG


__all__ = []


def create_detection_dataloaders(
    yaml_config_path,
    batch_size=64,
    cfg=DEFAULT_CFG,
):
    data = check_det_dataset(yaml_config_path)
    trainset, testset = data['train'], data.get('val') or data.get('test')
    train_loader = get_dataloader(trainset, data, cfg, batch_size=batch_size, rank=RANK, mode='train')
    test_loader = get_dataloader(testset, data, cfg, batch_size=batch_size * 2, rank=-1, mode='val')
    return {'train': train_loader, 'test': test_loader}



# DatasetParameters = namedtuple(
#     'DatasetParameters', ['num_classes', 'img_size', 'dataset_create_fn']
# )
# DATASET_WRAPPER_FNS = {
#     'coco': DatasetParameters(80, 416, create_coco_datasets),
#     'lisa': DatasetParameters(20, 416, create_lisa_datasets),
#     'voc': DatasetParameters(20, 448, create_voc_datasets),
#     'voc07': DatasetParameters(20, 448, create_voc07_datasets),
#     'voc_format_dataset': DatasetParameters(1, 320, create_voc_format_datasets),
#     'person_detection': DatasetParameters(1, 320, create_person_detection_datasets),
#     'custom_person_detection': DatasetParameters(
#         1, 640, create_person_detection_datasets
#     ),
#     'wider_face': DatasetParameters(1, 448, create_widerface_datasets),
#     'car_detection': DatasetParameters(1, 320, create_car_detection_datasets),
# }

# for dataset_name_key, dataset_parameters in DATASET_WRAPPER_FNS.items():
#     wrapper_fn_name = f'get_{dataset_name_key}_for_yolo'
#     wrapper_fn = make_dataset_wrapper(
#         wrapper_fn_name,
#         num_classes=dataset_parameters.num_classes,
#         img_size=dataset_parameters.img_size,
#         dataset_create_fn=dataset_parameters.dataset_create_fn,
#     )
#     globals()[wrapper_fn_name] = wrapper_fn
#     DATASET_WRAPPER_REGISTRY.register(dataset_name=dataset_name_key, model_type='yolo')(
#         wrapper_fn
#     )
#     __all__.append(wrapper_fn_name)
