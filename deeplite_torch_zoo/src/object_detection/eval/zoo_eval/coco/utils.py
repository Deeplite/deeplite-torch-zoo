def cxcywh_to_xywh(bbox):
    bbox[..., 0] -= bbox[..., 2] / 2
    bbox[..., 1] -= bbox[..., 3] / 2
    return bbox


def xywh_to_cxcywh(bbox):
    bbox[..., 0] += bbox[..., 2] / 2
    bbox[..., 1] += bbox[..., 3] / 2
    return bbox


def xywh_to_xyxy(bbox):
    bbox[..., 2] += bbox[..., 0]
    bbox[..., 3] += bbox[..., 1]
    return bbox
