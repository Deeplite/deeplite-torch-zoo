import pytest

from itertools import repeat
import torch.nn as nn

from deeplite_torch_zoo import get_model, get_zero_cost_estimator
from deeplite_torch_zoo.utils import init_seeds


REF_METRIC_VALUES = [
    ('fisher', 1.91525),
    ('grad_norm', 14.44768),
    ('grasp', -2.27878),
    ('jacob_cov', -32.233022),
    ('l2_norm', 172.978393),
    ('macs', 17794678784),
    ('nparams', 11220132),
    ('plain', 0.280363),
    ('snip', 317.93181),
    ('synflow', 3.31904e24),
    ('zico', 299.250910),
]


@pytest.mark.parametrize(
    ('metric_name', 'ref_value'),
    REF_METRIC_VALUES
)
def test_zero_cost_metrics_dataloader(metric_name, ref_value, cifar100_dataloaders, abs_tolerance=0.001):
    init_seeds(42)
    model = get_model(model_name='resnet18', dataset_name='cifar100', pretrained=False)
    loss = nn.CrossEntropyLoss()
    estimator_fn = get_zero_cost_estimator(metric_name=metric_name)
    metric_value = estimator_fn(model, dataloader=cifar100_dataloaders['test'], loss_fn=loss)
    assert pytest.approx(metric_value, abs_tolerance) == ref_value


@pytest.mark.parametrize(
    ('metric_name', 'ref_value'),
    REF_METRIC_VALUES
)
def test_zero_cost_metrics_generator(metric_name, ref_value, cifar100_dataloaders, abs_tolerance=0.001):
    init_seeds(42)
    model = get_model(model_name='resnet18', dataset_name='cifar100', pretrained=False)

    def data_generator(model, shuffle_data=True, input_gradient=False):
        test_loader = cifar100_dataloaders['test'] \
            if shuffle_data else repeat(next(iter(cifar100_dataloaders['test'])))
        for inputs, targets in test_loader:
            inputs.requires_grad_(input_gradient)
            outputs = model(inputs)
            yield inputs, outputs, targets, {'loss_kwarg': None}

    loss = nn.CrossEntropyLoss()
    def loss_fn(outputs, targets, **kwargs):
        assert kwargs['loss_kwarg'] is None
        return loss(outputs, targets)

    estimator_fn = get_zero_cost_estimator(metric_name=metric_name)
    metric_value = estimator_fn(model, model_output_generator=data_generator, loss_fn=loss_fn)
    assert pytest.approx(metric_value, abs_tolerance) == ref_value
