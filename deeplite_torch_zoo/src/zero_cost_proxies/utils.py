from typing import Iterable

import torch


def aggregate_statistic(value_array, reduction='sum'):
    if reduction is None:
        return value_array
    score = 0
    for item in value_array:
        if not isinstance(item, Iterable):
            score += float(item)
            continue
        item = torch.Tensor(item)
        if reduction == 'sum':
            score += float(torch.sum(item))
        elif reduction == 'channel_mean':
            score += float(
                torch.mean(torch.sum(item, dim=tuple(range(1, len(item.shape)))))
            )
        else:
            raise ValueError(
                f'`reduction` argument can be either `sum`, `channel_mean` or None, got {reduction}'
            )
    return score
