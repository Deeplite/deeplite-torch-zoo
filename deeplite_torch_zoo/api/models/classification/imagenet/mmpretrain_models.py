import torch
import torch.nn as nn  # pylint: disable=consider-using-from-import
import mmpretrain

from deeplite_torch_zoo.utils import LOGGER
from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY
from deeplite_torch_zoo.api.models.classification.imagenet.utils import NUM_IMAGENET_CLASSES
from deeplite_torch_zoo.api.models.classification.model_implementation_dict import MMPRETRAIN_BLACKLIST


class MMPretrainWrapper(nn.Module):
    def __init__(
            self,
            model_name,
            pretrained=False,
            num_classes=NUM_IMAGENET_CLASSES,
            dummy_input_size=(2, 3, 224, 224),
        ):
        super().__init__()
        self.model = mmpretrain.get_model(model_name, pretrained=pretrained, device='cpu')
        device = next(self.model.parameters()).device

        if hasattr(self.model.backbone, 'img_size'):
            dummy_input_size = (2, 3, *self.model.backbone.img_size)

        if num_classes != NUM_IMAGENET_CLASSES:
            head_type = self.model.head.__class__
            if hasattr(self.model.head, 'fc'):
                feature_dim = self.model.head.fc.in_features
            else:
                feature_dim = self.model.extract_feat(torch.randn(dummy_input_size).to(device)).shape[1]
            LOGGER.warning(f'Replacing MMPretrain model head with a {head_type} head with '\
                            'in_channels={feature_dim} and num_classes={num_classes}')
            self.model.head = head_type(in_channels=feature_dim, num_classes=num_classes).to(device)

    def forward(self, x):
        return self.model(x)


def make_wrapper_func(wrapper_fn_name, model_name_key):
    @MODEL_WRAPPER_REGISTRY.register(
        model_name=f'{model_name_key}_mmpretrain', dataset_name='imagenet', task_type='classification'
    )
    def wrapper_func(pretrained=False, num_classes=NUM_IMAGENET_CLASSES, **kwargs):
        model = MMPretrainWrapper(
            model_name_key,
            pretrained=pretrained,
            num_classes=num_classes,
            **kwargs,
        )
        return model

    wrapper_func.__name__ = wrapper_fn_name
    return wrapper_func


for model_name_tag in mmpretrain.list_models():
    if model_name_tag not in MMPRETRAIN_BLACKLIST:
        wrapper_name = '_'.join((model_name_tag, 'imagenet'))
        globals()[wrapper_name] = make_wrapper_func(wrapper_name, model_name_tag)
