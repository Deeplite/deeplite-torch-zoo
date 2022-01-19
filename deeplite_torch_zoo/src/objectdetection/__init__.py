import os
import sys
from pathlib import Path


path = sys.path
repo_path = str(Path(__file__).parent / 'ssd/repo/')

if repo_path not in os.environ.get('PYTHONPATH',''):
    path.insert(1, repo_path)
    os.environ["PYTHONPATH"] = os.pathsep.join(path)

