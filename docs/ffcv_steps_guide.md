## Open_cv installation from source

```
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

cd opencv
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=Release -D OPENCV_EXTRA_MODULES_PATH=opencv_contrib/modules -D CMAKE_INSTALL_PREFIX=/usr/local ..
make -j7
sudo make install
sudo ldconfig
pip3 install cupy-cuda113 numba
sudo apt-get install -y libturbojpeg0-dev
```

# Install ffcv using pip
```
pip3 install --upgrade pip
git clone https://github.com/libffcv/ffcv.git
pip3 install ffcv
```

# Additional repository to convert imagenet original data into required format
```
git clone https://github.com/libffcv/ffcv-imagenet.git

# Required environmental variables for the script:
export IMAGENET_DIR=/path/to/pytorch/format/imagenet/directory/
export WRITE_DIR=/your/path/here/

# Starting in the root of the Git repo:
cd examples;

# Serialize images with:
# - 500px side length maximum
# - 50% JPEG encoded, 90% raw pixel values
# - quality=90 JPEGs
./write_imagenet.sh 500 0.50 90

```

# Script to convert CIFAR100 dataset into ffcv formate

```
from argparse import ArgumentParser
from typing import List
import torchvision
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField

def main(train_dataset, val_dataset):
    datasets = {
        'train': torchvision.datasets.CIFAR100('/tmp', train=True, download=True),
        'test': torchvision.datasets.CIFAR100('/tmp', train=False, download=True)
        }

    for (name, ds) in datasets.items():
        path = train_dataset if name == 'train' else val_dataset
        writer = DatasetWriter(path, {
            'image': RGBImageField(),
            'label': IntField()
        })
        writer.from_indexed_dataset(ds)

main(train.ffcv, test.ffcv)
```