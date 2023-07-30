# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
import time
import re
import yaml
import urllib
import subprocess
import contextlib
import requests
import shutil
import platform
import glob
import zipfile

from tarfile import is_tarfile
from urllib import parse, request
from pathlib import Path
from zipfile import BadZipFile, ZipFile, is_zipfile

from tqdm import tqdm

import torch

from deeplite_torch_zoo.utils import LOGGER, is_dir_writeable, colorstr, ROOT, TQDM_BAR_FORMAT


MACOS, LINUX, WINDOWS = (platform.system() == x for x in ['Darwin', 'Linux', 'Windows'])  # environment booleans


def check_class_names(names):
    """Check class names. Map imagenet class codes to human-readable names if required. Convert lists to dicts."""
    if isinstance(names, list):  # names is a list
        names = dict(enumerate(names))  # convert to dict
    if isinstance(names, dict):
        # Convert 1) string keys to int, i.e. '0' to 0, and non-string values to strings, i.e. True to 'True'
        names = {int(k): str(v) for k, v in names.items()}
        n = len(names)
        if max(names.keys()) >= n:
            raise KeyError(f'{n}-class dataset requires class indices 0-{n - 1}, but you have invalid class indices '
                           f'{min(names.keys())}-{max(names.keys())} defined in your dataset YAML.')
        if isinstance(names[0], str) and names[0].startswith('n0'):  # imagenet class codes, i.e. 'n01440764'
            map = yaml_load(ROOT / 'cfg/datasets/ImageNet.yaml')['map']  # human-readable names
            names = {k: map[v] for k, v in names.items()}
    return names


def yaml_load(file='data.yaml', append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    with open(file, errors='ignore', encoding='utf-8') as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)

        # Add YAML filename to dict and return
        return {**yaml.safe_load(s), 'yaml_file': str(file)} if append_filename else yaml.safe_load(s)


def get_git_dir():
    """
    Determines whether the current file is part of a git repository and if so, returns the repository root directory.
    If the current file is not part of a git repository, returns None.

    Returns:
        (Path | None): Git root directory if found or None if not found.
    """
    for d in Path(__file__).parents:
        if (d / '.git').is_dir():
            return d
    return None  # no .git dir found


def get_datasets_dir():
    git_dir = get_git_dir()
    root = git_dir or Path()
    datasets_root = (root.parent if git_dir and is_dir_writeable(root.parent) else root).resolve()
    return datasets_root / 'datasets'


DATASETS_DIR = get_datasets_dir()


def emojis(string=''):
    """Return platform-dependent emoji-safe version of string."""
    return string.encode().decode('ascii', 'ignore') if WINDOWS else string


def is_online() -> bool:
    """
    Check internet connectivity by attempting to connect to a known online host.

    Returns:
        (bool): True if connection is successful, False otherwise.
    """
    import socket

    for host in '1.1.1.1', '8.8.8.8', '223.5.5.5':  # Cloudflare, Google, AliDNS:
        try:
            test_connection = socket.create_connection(address=(host, 53), timeout=2)
        except (socket.timeout, socket.gaierror, OSError):
            continue
        else:
            # If the connection was successful, close it to avoid a ResourceWarning
            test_connection.close()
            return True
    return False



def unzip_file(file, path=None, exclude=('.DS_Store', '__MACOSX'), exist_ok=False):
    """
    Unzips a *.zip file to the specified path, excluding files containing strings in the exclude list.

    If the zipfile does not contain a single top-level directory, the function will create a new
    directory with the same name as the zipfile (without the extension) to extract its contents.
    If a path is not provided, the function will use the parent directory of the zipfile as the default path.

    Args:
        file (str): The path to the zipfile to be extracted.
        path (str, optional): The path to extract the zipfile to. Defaults to None.
        exclude (tuple, optional): A tuple of filename strings to be excluded. Defaults to ('.DS_Store', '__MACOSX').
        exist_ok (bool, optional): Whether to overwrite existing contents if they exist. Defaults to False.

    Raises:
        BadZipFile: If the provided file does not exist or is not a valid zipfile.

    Returns:
        (Path): The path to the directory where the zipfile was extracted.
    """
    if not (Path(file).exists() and is_zipfile(file)):
        raise BadZipFile(f"File '{file}' does not exist or is a bad zip file.")
    if path is None:
        path = Path(file).parent  # default path

    # Unzip the file contents
    with ZipFile(file) as zipObj:
        file_list = [f for f in zipObj.namelist() if all(x not in f for x in exclude)]
        top_level_dirs = {Path(f).parts[0] for f in file_list}

        if len(top_level_dirs) > 1 or not file_list[0].endswith('/'):
            path = Path(path) / Path(file).stem  # define new unzip directory

        # Check if destination directory already exists and contains files
        extract_path = Path(path) / list(top_level_dirs)[0]
        if extract_path.exists() and any(extract_path.iterdir()) and not exist_ok:
            # If it exists and is not empty, return the path without unzipping
            LOGGER.info(f'Skipping {file} unzip (already unzipped)')
            return path

        for f in file_list:
            zipObj.extract(f, path=path)

    return path  # return unzip dir


def check_disk_space(url='https://ultralytics.com/assets/coco128.zip', sf=1.5, hard=True):
    """
    Check if there is sufficient disk space to download and store a file.

    Args:
        url (str, optional): The URL to the file. Defaults to 'https://ultralytics.com/assets/coco128.zip'.
        sf (float, optional): Safety factor, the multiplier for the required free space. Defaults to 2.0.
        hard (bool, optional): Whether to throw an error or not on insufficient disk space. Defaults to True.

    Returns:
        (bool): True if there is sufficient disk space, False otherwise.
    """
    with contextlib.suppress(Exception):
        gib = 1 << 30  # bytes per GiB
        data = int(requests.head(url).headers['Content-Length']) / gib  # file size (GB)
        total, used, free = (x / gib for x in shutil.disk_usage('/'))  # bytes
        if data * sf < free:
            return True  # sufficient space

        # Insufficient space
        text = (f'WARNING ⚠️ Insufficient free disk space {free:.1f} GB < {data * sf:.3f} GB required, '
                f'Please free {data * sf - free:.1f} GB additional disk space and try again.')
        if hard:
            raise MemoryError(text)
        else:
            LOGGER.warning(text)
            return False

            # Pass if error
    return True


def safe_download(url,
                  file=None,
                  dir=None,
                  unzip=True,
                  delete=False,
                  curl=False,
                  retry=3,
                  min_bytes=1E0,
                  progress=True):
    """
    Downloads files from a URL, with options for retrying, unzipping, and deleting the downloaded file.

    Args:
        url (str): The URL of the file to be downloaded.
        file (str, optional): The filename of the downloaded file.
            If not provided, the file will be saved with the same name as the URL.
        dir (str, optional): The directory to save the downloaded file.
            If not provided, the file will be saved in the current working directory.
        unzip (bool, optional): Whether to unzip the downloaded file. Default: True.
        delete (bool, optional): Whether to delete the downloaded file after unzipping. Default: False.
        curl (bool, optional): Whether to use curl command line tool for downloading. Default: False.
        retry (int, optional): The number of times to retry the download in case of failure. Default: 3.
        min_bytes (float, optional): The minimum number of bytes that the downloaded file should have, to be considered
            a successful download. Default: 1E0.
        progress (bool, optional): Whether to display a progress bar during the download. Default: True.
    """
    f = dir / url2file(url) if dir else Path(file)  # URL converted to filename
    if '://' not in str(url) and Path(url).is_file():  # URL exists ('://' check required in Windows Python<3.10)
        f = Path(url)  # filename
    elif not f.is_file():  # URL and file do not exist
        assert dir or file, 'dir or file required for download'
        f = dir / url2file(url) if dir else Path(file)
        desc = f"Downloading {clean_url(url)} to '{f}'"
        LOGGER.info(f'{desc}...')
        f.parent.mkdir(parents=True, exist_ok=True)  # make directory if missing
        check_disk_space(url)
        for i in range(retry + 1):
            try:
                if curl or i > 0:  # curl download with retry, continue
                    s = 'sS' * (not progress)  # silent
                    r = subprocess.run(['curl', '-#', f'-{s}L', url, '-o', f, '--retry', '3', '-C', '-']).returncode
                    assert r == 0, f'Curl return value {r}'
                else:  # urllib download
                    method = 'torch'
                    if method == 'torch':
                        torch.hub.download_url_to_file(url, f, progress=progress)
                    else:
                        with request.urlopen(url) as response, tqdm(total=int(response.getheader('Content-Length', 0)),
                                                                    desc=desc,
                                                                    disable=not progress,
                                                                    unit='B',
                                                                    unit_scale=True,
                                                                    unit_divisor=1024,
                                                                    bar_format=TQDM_BAR_FORMAT) as pbar:
                            with open(f, 'wb') as f_opened:
                                for data in response:
                                    f_opened.write(data)
                                    pbar.update(len(data))

                if f.exists():
                    if f.stat().st_size > min_bytes:
                        break  # success
                    f.unlink()  # remove partial downloads
            except Exception as e:
                if i == 0 and not is_online():
                    raise ConnectionError(emojis(f'❌  Download failure for {url}. Environment is not online.')) from e
                elif i >= retry:
                    raise ConnectionError(emojis(f'❌  Download failure for {url}. Retry limit reached.')) from e
                LOGGER.warning(f'⚠️ Download failure, retrying {i + 1}/{retry} {url}...')

    if unzip and f.exists() and f.suffix in ('', '.zip', '.tar', '.gz'):
        unzip_dir = dir or f.parent  # unzip to dir if provided else unzip in place
        LOGGER.info(f'Unzipping {f} to {unzip_dir.absolute()}...')
        if is_zipfile(f):
            unzip_dir = unzip_file(file=f, path=unzip_dir)  # unzip
        elif f.suffix == '.tar':
            subprocess.run(['tar', 'xf', f, '--directory', unzip_dir], check=True)  # unzip
        elif f.suffix == '.gz':
            subprocess.run(['tar', 'xfz', f, '--directory', unzip_dir], check=True)  # unzip
        if delete:
            f.unlink()  # remove zip
        return unzip_dir


def url2file(url):
    """Convert URL to filename, i.e. https://url.com/file.txt?auth -> file.txt."""
    return Path(clean_url(url)).name


def clean_url(url):
    """Strip auth from URL, i.e. https://url.com/file.txt?auth -> https://url.com/file.txt."""
    url = str(Path(url)).replace(':/', '://')  # Pathlib turns :// -> :/
    return urllib.parse.unquote(url).split('?')[0]  # '%2F' to '/', split https://url.com/file.txt?auth


def check_suffix(file='yolov8n.pt', suffix='.pt', msg=''):
    """Check file(s) for acceptable suffix."""
    if file and suffix:
        if isinstance(suffix, str):
            suffix = (suffix, )
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower().strip()  # file suffix
            if len(s):
                assert s in suffix, f'{msg}{f} acceptable suffix is {suffix}, not {s}'


def check_file(file, suffix='', download=True, hard=True):
    """Search/download file (if necessary) and return path."""
    check_suffix(file, suffix)  # optional
    file = str(file).strip()  # convert to string and strip spaces
    if not file or ('://' not in file and Path(file).exists()):  # exists ('://' check required in Windows Python<3.10)
        return file
    elif download and file.lower().startswith(('https://', 'http://', 'rtsp://', 'rtmp://')):  # download
        url = file  # warning: Pathlib turns :// -> :/
        file = url2file(file)  # '%2F' to '/', split https://url.com/file.txt?auth
        if Path(file).exists():
            LOGGER.info(f'Found {clean_url(url)} locally at {file}')  # file already exists
        else:
            safe_download(url=url, file=file, unzip=False)
        return file
    else:  # search
        files = glob.glob(str(ROOT / 'cfg' / '**' / file), recursive=True)  # find file
        if not files and hard:
            raise FileNotFoundError(f"'{file}' does not exist")
        elif len(files) > 1 and hard:
            raise FileNotFoundError(f"Multiple files match '{file}', specify exact path: {files}")
        return files[0] if len(files) else []  # return file


def check_det_dataset(dataset, autodownload=True):
    """Download, check and/or unzip dataset if not found locally."""
    data = check_file(dataset)

    # Download (optional)
    extract_dir = ''
    if isinstance(data, (str, Path)) and (zipfile.is_zipfile(data) or is_tarfile(data)):
        new_dir = safe_download(data, dir=DATASETS_DIR, unzip=True, delete=False, curl=False)
        data = next((DATASETS_DIR / new_dir).rglob('*.yaml'))
        extract_dir, autodownload = data.parent, False

    # Read yaml (optional)
    if isinstance(data, (str, Path)):
        data = yaml_load(data, append_filename=True)  # dictionary

    # Checks
    for k in 'train', 'val':
        if k not in data:
            raise SyntaxError(
                emojis(f"{dataset} '{k}:' key missing ❌.\n'train' and 'val' are required in all data YAMLs."))
    if 'names' not in data and 'nc' not in data:
        raise SyntaxError(emojis(f"{dataset} key missing ❌.\n either 'names' or 'nc' are required in all data YAMLs."))
    if 'names' in data and 'nc' in data and len(data['names']) != data['nc']:
        raise SyntaxError(emojis(f"{dataset} 'names' length {len(data['names'])} and 'nc: {data['nc']}' must match."))
    if 'names' not in data:
        data['names'] = [f'class_{i}' for i in range(data['nc'])]
    else:
        data['nc'] = len(data['names'])

    data['names'] = check_class_names(data['names'])

    # Resolve paths
    path = Path(extract_dir or data.get('path') or Path(data.get('yaml_file', '')).parent)  # dataset root

    if not path.is_absolute():
        path = (DATASETS_DIR / path).resolve()
    data['path'] = path  # download scripts
    for k in 'train', 'val', 'test':
        if data.get(k):  # prepend path
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith('../'):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]

    # Parse yaml
    train, val, test, s = (data.get(x) for x in ('train', 'val', 'test', 'download'))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exists() for x in val):
            name = clean_url(dataset)  # dataset name with URL auth stripped
            m = f"\nDataset '{name}' images not found ⚠️, missing paths %s" % [str(x) for x in val if not x.exists()]
            if s and autodownload:
                LOGGER.warning(m)
            else:
                m += f"\nNote dataset download directory is '{DATASETS_DIR}'"
                raise FileNotFoundError(m)
            t = time.time()
            if s.startswith('http') and s.endswith('.zip'):  # URL
                safe_download(url=s, dir=DATASETS_DIR, delete=True)
                r = None  # success
            elif s.startswith('bash '):  # bash script
                LOGGER.info(f'Running {s} ...')
                r = os.system(s)
            else:  # python script
                r = exec(s, {'yaml': data})  # return None
            dt = f'({round(time.time() - t, 1)}s)'
            s = f"success ✅ {dt}, saved to {colorstr('bold', DATASETS_DIR)}" if r in (0, None) else f'failure {dt} ❌'
            LOGGER.info(f'Dataset download {s}\n')
    return data  # dictionary
