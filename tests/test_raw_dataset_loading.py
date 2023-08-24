import shutil
from pathlib import Path

import pytest

from deeplite_torch_zoo import get_dataloaders


DATASETS_ROOT = Path('/public-shared/datasets/RAW_DATASETS/custom_data/pascal_cropped_clf/raw')
BATCH_SIZE = 128


@pytest.mark.parametrize(
    ('dataset_name', 'tmp_dataset_folder', 'train_dataloader_len', 'test_dataloader_len'),
    [
        ('pascalraw', DATASETS_ROOT, 40, 11),
    ],
)
def test_get_dataloaders(dataset_name, tmp_dataset_folder):
    p = Path(tmp_dataset_folder)
    dataloaders = get_dataloaders(
        data_root=tmp_dataset_folder,
        dataset_name=dataset_name,
        batch_size=BATCH_SIZE,
        test_batch_size=BATCH_SIZE,
    )
    assert len(dataloaders['train']) == train_dataloader_len
    assert len(dataloaders['test']) == test_dataloader_len
