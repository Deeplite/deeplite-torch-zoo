from .FPN import PyramidFeatures as FPN
from .PAN import PAN

__all__ = ['build_neck']

NECK_MAP = {
    'FPN': FPN,
    'PAN': PAN,
}


def build_neck(neck_name, **kwargs):
    if neck_name not in NECK_MAP:
        raise ValueError(f'Neck {neck_name} not supported. Supported neck types: {NECK_MAP.keys()}')
    neck = NECK_MAP[neck_name](**kwargs)
    return neck
