from setuptools import find_packages, setup
import os, pathlib
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.sdist import sdist
from setuptools import setup, find_packages
from subprocess import check_call


# Package
HERE = pathlib.Path(__file__).parent

INSTALL_REQUIRES = [
    "numpy==1.18.5",
    "torch==1.4.0",
    "torchvision==0.5.0",
    "opencv-python",
    "scipy>=1.4.1",
    "pycocotools",
    "Cython==0.28.4",
    "scikit-image==0.15.0",
    "tqdm==4.46.0",
    "albumentations==0.1.8",
    "pretrainedmodels==0.7.4",
    "torchfcn",
    "tensorboardX",
    "mmcv==1.2.0",
    "json-tricks>=3.15.4",
    "poseval@git+https://github.com/svenkreiss/poseval.git#egg=poseval-0.1.0",
    "black",
    "isort",
]


def create_init_files_in_submodules():
    submodules_init = [
        "deeplite_torch_zoo/src/objectdetection/mb_ssd/repo/__init__.py",
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
    version="1.0.0",
    description="deeplite Torch Zoo",
    long_description="The deeplite-torch-zoo package is a collection of popular model architectures and their datasets for deep learning for pytorch framework.",
    author="Deeplite",
    author_email="info@deeplite.ai",
    url="https://www.deeplite.ai",
    license=license,
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
        "Programming Language:: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Operating System :: POSIX :: Linux",
        "Natural Language :: English",
        "License :: Other/Proprietary License",
        "Environment :: Console",
    ],
    keywords="deep_neural_network deep_learning zoo model datasets pytorch deeplite",
)
