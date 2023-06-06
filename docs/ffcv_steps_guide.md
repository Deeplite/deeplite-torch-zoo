## Open_cv installation from source

```
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

cd opencv
mkdir build
cd build
cmake -DOPENCV_EXTRA_MODULES_PATH=opencv_contrib/modules 
      -DWITH_CUDA=ON
      -DCUDA_ARCH_BIN=7.5,8.0,8.6
      -DCMAKE_BUILD_TYPE=RELEASE
      -DOPENCV_GENERATE_PKGCONFIG=YES
      -DCMAKE_INSTALL_PREFIX=/usr/local
      ..
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