import shutil
from pathlib import Path

import pytest

from deeplite_torch_zoo import get_dataloaders


DATASETS_ROOT = Path('/neutrino/datasets/')
BATCH_SIZE = 128


@pytest.mark.parametrize(
    ('dataset_name', 'tmp_dataset_files', 'tmp_dataset_folders',
     'train_dataloader_len', 'test_dataloader_len'),
    [
        ('cifar100', ('cifar-100-python.tar.gz', ), ('cifar-100-python', ), 391, 79),
        ('cifar10', ('cifar-10-python.tar.gz', ), ('cifar-10-batches-py', ), 391, 79),
        ('imagenette', ('imagenette.zip', ), ('imagenette', ), 74, 31),
        ('imagewoof', ('imagewoof.zip', ), ('imagewoof', ), 71, 31),
        ('mnist', (), ('MNIST', ), 469, 79),
    ],
)
def test_get_dataloaders(dataset_name, tmp_dataset_files, tmp_dataset_folders,
                         train_dataloader_len, test_dataloader_len,
                         data_root='./'):
    p = Path(data_root)
    dataloaders = get_dataloaders(
        data_root=data_root,
        dataset_name=dataset_name,
        batch_size=BATCH_SIZE
    )
    assert len(dataloaders['train']) == train_dataloader_len
    assert len(dataloaders['test']) == test_dataloader_len
    for file in tmp_dataset_files:
        (p / file).unlink()
    for folder in tmp_dataset_folders:
        shutil.rmtree(p / folder)


@pytest.mark.parametrize(
    ('dataset_name', 'data_root', 'train_dataloader_len', 'test_dataloader_len'),
    [
        ('vww', str(DATASETS_ROOT / 'vww'), 901, 63),
        ('imagenet', str(DATASETS_ROOT / 'imagenet16'), 1408, 332),
        ('imagenet', str(DATASETS_ROOT / 'imagenet10'), 3010, 118),
        ('imagenet', str(DATASETS_ROOT / 'imagenet'), 10010, 391),
        ('voc', str(DATASETS_ROOT / 'VOCdevkit'), 130, 39),
        ('coco', str(DATASETS_ROOT / 'coco'), 11829, 500),
    ],
)
@pytest.mark.local
def test_get_dataloaders_local(dataset_name, data_root, train_dataloader_len, test_dataloader_len):
    dataloaders = get_dataloaders(
        data_root=data_root,
        dataset_name=dataset_name,
        batch_size=BATCH_SIZE
    )
    assert len(dataloaders['train']) == train_dataloader_len
    assert len(dataloaders['test']) == test_dataloader_len
