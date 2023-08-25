# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.yolo.engine.model import YOLO
from ultralytics.yolo.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, callbacks
from deeplite_torch_zoo import get_model


def patched_init(obj, model_name, num_classes=None,
                 task=None, session=None, pretrained=False, pretraining_dataset='coco'):
    obj.callbacks = callbacks.get_default_callbacks()
    obj.predictor = None  # reuse predictor
    obj.model = None  # model object
    obj.trainer = None  # trainer object
    obj.task = task if task is not None else 'detect' # task type
    obj.ckpt = True  # if loaded from *.pt
    obj.cfg = None  # if loaded from *.yaml
    obj.ckpt_path = None
    obj.overrides = {}  # overrides for trainer object
    obj.metrics = None  # validation/training metrics
    obj.session = session  # HUB session
    obj.num_classes = num_classes

    obj.cfg = model_name
    obj.model = get_model(
        model_name=model_name,
        dataset_name=pretraining_dataset,
        pretrained=pretrained,
        num_classes=num_classes,
        custom_head='yolo8',  # can only work with v8 head as of now
    )
    obj.overrides['model'] = obj.cfg

    # Below added to allow export from yamls
    args = {**DEFAULT_CFG_DICT, **obj.overrides}  # combine model and default args, preferring model args
    obj.model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
    obj.model.task = obj.task
    obj.model.model_name = model_name


YOLO.__init__ = patched_init
