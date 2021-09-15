import os
import sys

from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent


path = sys.path
MISSING = False
if "PYTHONPATH" not in os.environ:
	os.environ["PYTHONPATH"] = ""
if 'src/objectdetection/ssd/repo/' not in os.environ["PYTHONPATH"]:
    path.insert(1, str(get_project_root() / 'deeplite_torch_zoo/src/objectdetection/ssd/repo/'))
    MISSING = True
if 'src/segmentation/deeplab/repo/' not in os.environ["PYTHONPATH"]:
    path.insert(1, str(get_project_root() / 'deeplite_torch_zoo/src/segmentation/deeplab/repo/'))
    MISSING = True

if MISSING:
	os.environ["PYTHONPATH"] = os.pathsep.join(path)
