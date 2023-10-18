import timm
import torch.nn as nn

from deeplite_torch_zoo.utils import LOGGER


class TimmWrapperBackbone(nn.Module):
    """
    Wrapper to use backbones from timm
    https://github.com/huggingface/pytorch-image-models
    """

    def __init__(
        self,
        model_name,
        features_only=True,
        pretrained=False,
        checkpoint_path=None,
        in_channels=3,
        feature_map_indices=(-3, -2, -1),
        **kwargs,
    ):
        super(TimmWrapperBackbone, self).__init__()
        self.backbone = timm.create_model(
            model_name=model_name,
            features_only=features_only,
            pretrained=pretrained,
            in_chans=in_channels,
            checkpoint_path=checkpoint_path,
            **kwargs,
        )
        self.feature_map_indices = feature_map_indices

        # Remove unused layers
        self.backbone.global_pool = None
        self.backbone.fc = None
        self.backbone.classifier = None

        feature_info = getattr(self.backbone, 'feature_info', None)
        if feature_info:
            LOGGER.info(f'timm backbone feature channels: {feature_info.channels()}')
        self.out_shape = [feature_info.channels()[idx] for idx in self.feature_map_indices]

    def forward(self, x):
        outs = self.backbone(x)
        if isinstance(outs, (list, tuple)):
            features = tuple(outs)
        else:
            features = (outs,)
        return tuple(features[idx] for idx in self.feature_map_indices)
