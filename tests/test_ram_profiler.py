import pandas as pd
import pytest
import torch
import torch.nn as nn

from deeplite_torch_zoo.utils.profiler import profile_ram, ram_report


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(32 * 32 * 32, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


@pytest.mark.parametrize(
    (
        'columns',
        'num_nodes',
        'peak_ram',
        'input_shape_0',
        'scope_1',
        'output_shape_3',
    ),
    [
        (
            ['weight', 'bias', 'input_shape', 'output_shape', 'in_tensors',
             'out_tensors', 'active_blocks', 'ram', 'scope'],
            8,
            262144,
            [1, 3, 32, 32],
            'relu1',
            [1, 32, 32, 32],
        )
    ]
)
def test_ram_profiling(
        columns,
        num_nodes,
        peak_ram,
        input_shape_0,
        scope_1,
        output_shape_3,
    ):
    model = DummyModel()
    input_tensor = torch.randn(1, 3, 32, 32)
    nodes_info = profile_ram(model, input_tensor, detailed=True)

    assert isinstance(nodes_info, pd.DataFrame)
    assert all(column in nodes_info.columns for column in columns)
    assert len(nodes_info) == num_nodes
    assert nodes_info.ram.max() == peak_ram
    assert nodes_info.iloc[0].input_shape[0] == input_shape_0
    assert nodes_info.iloc[1].scope == scope_1
    assert nodes_info.iloc[3].output_shape[0] == output_shape_3
