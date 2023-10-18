import pathlib
from setuptools import find_packages, setup


with open('LICENSE.md') as f:
    license = f.read()

with open('README.md') as f:
    long_description = f.read()


# Package
HERE = pathlib.Path(__file__).parent

INSTALL_REQUIRES = [
    'setuptools<65.6.0',
    'urllib3==1.26.6',
    'torch>=1.4, <=2.0.1',
    'opencv-python<=4.6.0.66',
    'scipy>=1.4.1',
    'numpy==1.20.0',
    'Cython==0.29.30',
    'tqdm>=4.46.0',
    'albumentations==1.0.3',
    'tensorboardX==2.4.1',
    'pyvww==0.1.1',
    'timm==0.9.2',
    'pytorchcv==0.0.67',
    'texttable==1.6.4',
    'torchprofile==0.0.4',
    'addict==2.4.0',
    'Wand==0.6.11',
    'pytz',
    'pandas',
    'ultralytics==8.0.107',
    'tensorboard>=2.11.2',
]

setup(
    name='deeplite-torch-zoo',
    version='2.0.4',
    description='The deeplite-torch-zoo package is a collection of popular pretrained deep learning models and their datasets for PyTorch framework.',
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
