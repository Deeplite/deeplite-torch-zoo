import os
import pathlib
from subprocess import check_call

from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.sdist import sdist

with open('LICENSE.md') as f:
    license = f.read()

with open('README.md') as f:
    long_description = f.read()


# Package
HERE = pathlib.Path(__file__).parent

INSTALL_REQUIRES = [
    "torch>=1.4, <=1.8.1",
    "opencv-python",
    "scipy>=1.4.1",
    "numpy==1.19.5",
    "pycocotools==2.0.4",
    "Cython==0.29.30",
    "tqdm==4.46.0",
    "albumentations",
    "pretrainedmodels==0.7.4",
    "torchfcn==1.9.7",
    "tensorboardX==2.4.1",
    "pyvww==0.1.1",
    "timm==0.5.4",
    "texttable==1.6.4",
    "pytz",
    "torchmetrics==0.8.0",
    "ptflops==0.6.2",
    "pandas",
]


def create_init_files_in_submodules():
    submodules_init = [
        "deeplite_torch_zoo/src/objectdetection/ssd/repo/__init__.py",
        "deeplite_torch_zoo/src/segmentation/deeplab/repo/__init__.py",
        "deeplite_torch_zoo/src/segmentation/unet_scse/repo/__init__.py",
        "deeplite_torch_zoo/src/segmentation/unet_scse/repo/src/losses/__init__.py"
    ]
    for _f in submodules_init:
        if not os.path.exists(_f):
            with open(_f, 'w'):
                pass

def gitcmd_update_submodules():
    ''' Check if the package is being deployed as a git repository. If so, recursively
        update all dependencies.

        @returns True if the package is a git repository and the modules were updated.
            False otherwise.
    '''
    if os.path.exists(os.path.join(HERE, '.git')):
        check_call(['git', 'submodule', 'update', '--init', '--recursive'])
        return True

    return False


class gitcmd_develop(develop):
    ''' Specialized packaging class that runs git submodule update --init --recursive
        as part of the update/install procedure.
    '''
    def run(self):
        gitcmd_update_submodules()
        develop.run(self)


class gitcmd_install(install):
    ''' Specialized packaging class that runs git submodule update --init --recursive
        as part of the update/install procedure.
    '''
    def run(self):
        gitcmd_update_submodules()
        create_init_files_in_submodules()
        install.run(self)


class gitcmd_sdist(sdist):
    ''' Specialized packaging class that runs git submodule update --init --recursive
        as part of the update/install procedure;.
    '''
    def run(self):
        gitcmd_update_submodules()
        sdist.run(self)

setup(
    cmdclass={
        'develop': gitcmd_develop,
        'install': gitcmd_install,
        'sdist': gitcmd_sdist,
    },
    name="deeplite-torch-zoo",
    version="1.2.7",
    description="The deeplite-torch-zoo package is a collection of popular pretrained deep learning models and their datasets for PyTorch framework.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Multi-licensing",
    author="Deeplite",
    author_email="support@deeplite.ai",
    url="https://github.com/Deeplite/deeplite-torch-zoo",
    include_package_data=True,
    packages=find_packages(exclude=["tests*"]),
    tests_require=['pytest', 'pylint'],
    setup_requires=['pytest-runner', 'pytest-pylint'],
    install_requires=INSTALL_REQUIRES,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Operating System :: POSIX :: Linux",
        "Natural Language :: English",
        "License :: Other/Proprietary License",
        "Environment :: Console",
    ],
    keywords="deep_neural_network deep_learning zoo model datasets pytorch deeplite",
)
