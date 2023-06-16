import sys
import pathlib
import atexit

from setuptools import find_packages, setup
from setuptools.command.install import install


with open('LICENSE.md') as f:
    license = f.read()

with open('README.md') as f:
    long_description = f.read()


# Package
HERE = pathlib.Path(__file__).parent

INSTALL_REQUIRES = [
    'setuptools<65.6.0',
    'urllib3==1.26.6',
    'torch>=1.4, <=2.0.0',
    'opencv-python<=4.6.0.66',
    'scipy>=1.4.1',
    'numpy==1.20.0',
    'pycocotools==2.0.4',
    'Cython==0.29.30',
    'tqdm>=4.46.0',
    'albumentations==1.0.3',
    'tensorboardX==2.4.1',
    'pyvww==0.1.1',
    'timm==0.5.4',
    'pytorchcv==0.0.67',
    'texttable==1.6.4',
    'torchprofile==0.0.4',
    'addict==2.4.0',
    'Wand==0.6.11',
    'pytz==2023.3',
    'pandas==1.4.4',
    'ultralytics==8.0.107',
    'tensorboard>=2.11.2',
    'openmim==0.3.7',
]


python_version = sys.version_info
if python_version < (3, 7, 0):
    INSTALL_REQUIRES.append('torchinfo==1.5.4')
else:
    INSTALL_REQUIRES.append('torchinfo==1.7.2')


class PostInstall(install):
    def run(self):
        def _post_install():
            import mim
            mim.install(['mmpretrain>=1.0.0rc8',])
        atexit.register(_post_install)
        install.run(self)


setup(
    name='deeplite-torch-zoo',
    version='2.0.0',
    description='Deeplite Torch Zoo is a collection of state of the art deep learning architectures, ' \
        'training pipelines and pre-trained models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='Multi-licensing',
    author='Deeplite',
    author_email='support@deeplite.ai',
    url='https://github.com/Deeplite/deeplite-torch-zoo',
    include_package_data=True,
    packages=find_packages(exclude=['tests*']),
    tests_require=['pytest', 'pylint'],
    setup_requires=['pytest-runner', 'pytest-pylint'],
    install_requires=INSTALL_REQUIRES,
    cmdclass={'install': PostInstall},
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Operating System :: POSIX :: Linux',
        'Natural Language :: English',
        'License :: Other/Proprietary License',
        'Environment :: Console',
    ],
    keywords='deep_neural_network deep_learning zoo model datasets pytorch deeplite',
)