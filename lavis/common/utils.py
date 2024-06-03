"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import io
import json
import logging
import os
import pickle
import re
import torch
import shutil
import urllib
import urllib.error
import urllib.request
from typing import Optional
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import yaml
from iopath.common.download import download
from iopath.common.file_io import file_lock, g_pathmgr
from lavis.common.registry import registry
from torch.utils.model_zoo import tqdm
from torchvision.datasets.utils import (
    check_integrity,
    download_file_from_google_drive,
    extract_archive,
)

from functools import partial

import operator
import importlib
from typing import Callable, Optional
import pkg_resources
from packaging.version import Version

TQDM_ARGS = {"bar_format": "{l_bar}{bar:10}{r_bar}", "colour": "green"}


def compare_version(package: str, op: Callable, version: str, use_base_version: bool = False) -> bool:
    """Compare package version with some requirements.

    >>> compare_version("torch", operator.ge, "0.1")
    True
    >>> compare_version("does_not_exist", operator.ge, "0.0")
    False

    """
    try:
        pkg = importlib.import_module(package)
    except (ImportError, pkg_resources.DistributionNotFound):
        return False
    try:
        if hasattr(pkg, "__version__"):
            pkg_version = Version(pkg.__version__)
        else:
            # try pkg_resources to infer version
            pkg_version = Version(pkg_resources.get_distribution(package).version)
    except TypeError:
        # this is mocked by Sphinx, so it should return True to generate all summaries
        return True
    if use_base_version:
        pkg_version = Version(pkg_version.base_version)
    return op(pkg_version, Version(version))


_TORCH_GREATER_EQUAL_2_0 = compare_version("torch", operator.ge, "2.0.0")
_TORCH_GREATER_EQUAL_1_12 = compare_version("torch", operator.ge, "1.12.0")
if _TORCH_GREATER_EQUAL_1_12:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
else:
    checkpoint_wrapper = None


def now():
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d%H%M")[:-1]


def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def get_cache_path(rel_path):
    return os.path.expanduser(os.path.join(registry.get_path("cache_root"), rel_path))


def get_abs_path(rel_path):
    return os.path.join(registry.get_path("library_root"), rel_path)


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


# The following are adapted from torchvision and vissl
# torchvision: https://github.com/pytorch/vision
# vissl: https://github.com/facebookresearch/vissl/blob/main/vissl/utils/download.py


def makedir(dir_path):
    """
    Create the directory if it does not exist.
    """
    is_success = False
    try:
        if not g_pathmgr.exists(dir_path):
            g_pathmgr.mkdirs(dir_path)
        is_success = True
    except BaseException:
        print(f"Error creating directory: {dir_path}")
    return is_success


def get_redirected_url(url: str):
    """
    Given a URL, returns the URL it redirects to or the
    original URL in case of no indirection
    """
    import requests

    with requests.Session() as session:
        with session.get(url, stream=True, allow_redirects=True) as response:
            if response.history:
                return response.url
            else:
                return url


def to_google_drive_download_url(view_url: str) -> str:
    """
    Utility function to transform a view URL of google drive
    to a download URL for google drive
    Example input:
        https://drive.google.com/file/d/137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp/view
    Example output:
        https://drive.google.com/uc?export=download&id=137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp
    """
    splits = view_url.split("/")
    assert splits[-1] == "view"
    file_id = splits[-2]
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def download_google_drive_url(url: str, output_path: str, output_file_name: str):
    """
    Download a file from google drive
    Downloading an URL from google drive requires confirmation when
    the file of the size is too big (google drive notifies that
    anti-viral checks cannot be performed on such files)
    """
    import requests

    with requests.Session() as session:
        # First get the confirmation token and append it to the URL
        with session.get(url, stream=True, allow_redirects=True) as response:
            for k, v in response.cookies.items():
                if k.startswith("download_warning"):
                    url = url + "&confirm=" + v

        # Then download the content of the file
        with session.get(url, stream=True, verify=True) as response:
            makedir(output_path)
            path = os.path.join(output_path, output_file_name)
            total_size = int(response.headers.get("Content-length", 0))
            with open(path, "wb") as file:
                from tqdm import tqdm

                with tqdm(total=total_size) as progress_bar:
                    for block in response.iter_content(chunk_size=io.DEFAULT_BUFFER_SIZE):
                        file.write(block)
                        progress_bar.update(len(block))


def _get_google_drive_file_id(url: str) -> Optional[str]:
    parts = urlparse(url)

    if re.match(r"(drive|docs)[.]google[.]com", parts.netloc) is None:
        return None

    match = re.match(r"/file/d/(?P<id>[^/]*)", parts.path)
    if match is None:
        return None

    return match.group("id")


def _urlretrieve(url: str, filename: str, chunk_size: int = 1024) -> None:
    with open(filename, "wb") as fh:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "vissl"})) as response:
            with tqdm(total=response.length) as pbar:
                for chunk in iter(lambda: response.read(chunk_size), ""):
                    if not chunk:
                        break
                    pbar.update(chunk_size)
                    fh.write(chunk)


def download_url(
    url: str,
    root: str,
    filename: Optional[str] = None,
    md5: Optional[str] = None,
) -> None:
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under.
                                  If None, use the basename of the URL.
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    makedir(root)

    # check if file is already present locally
    if check_integrity(fpath, md5):
        print("Using downloaded and verified file: " + fpath)
        return

    # expand redirect chain if needed
    url = get_redirected_url(url)

    # check if file is located on Google Drive
    file_id = _get_google_drive_file_id(url)
    if file_id is not None:
        return download_file_from_google_drive(file_id, root, filename, md5)

    # download the file
    try:
        print("Downloading " + url + " to " + fpath)
        _urlretrieve(url, fpath)
    except (urllib.error.URLError, IOError) as e:  # type: ignore[attr-defined]
        if url[:5] == "https":
            url = url.replace("https:", "http:")
            print("Failed download. Trying https -> http instead." " Downloading " + url + " to " + fpath)
            _urlretrieve(url, fpath)
        else:
            raise e

    # check integrity of downloaded file
    if not check_integrity(fpath, md5):
        raise RuntimeError("File not found or corrupted.")


def download_and_extract_archive(
    url: str,
    download_root: str,
    extract_root: Optional[str] = None,
    filename: Optional[str] = None,
    md5: Optional[str] = None,
    remove_finished: bool = False,
) -> None:
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print("Extracting {} to {}".format(archive, extract_root))
    extract_archive(archive, extract_root, remove_finished)


def cache_url(url: str, cache_dir: str) -> str:
    """
    This implementation downloads the remote resource and caches it locally.
    The resource will only be downloaded if not previously requested.
    """
    parsed_url = urlparse(url)
    dirname = os.path.join(cache_dir, os.path.dirname(parsed_url.path.lstrip("/")))
    makedir(dirname)
    filename = url.split("/")[-1]
    cached = os.path.join(dirname, filename)
    with file_lock(cached):
        if not os.path.isfile(cached):
            logging.info(f"Downloading {url} to {cached} ...")
            cached = download(url, dirname, filename=filename)
    logging.info(f"URL {url} cached in {cached}")
    return cached


# TODO (prigoyal): convert this into RAII-style API
def create_file_symlink(file1, file2):
    """
    Simply create the symlinks for a given file1 to file2.
    Useful during model checkpointing to symlinks to the
    latest successful checkpoint.
    """
    try:
        if g_pathmgr.exists(file2):
            g_pathmgr.rm(file2)
        g_pathmgr.symlink(file1, file2)
    except Exception as e:
        logging.info(f"Could NOT create symlink. Error: {e}")


def save_file(data, filename, append_to_json=True, verbose=True):
    """
    Common i/o utility to handle saving data to various file formats.
    Supported:
        .pkl, .pickle, .npy, .json
    Specifically for .json, users have the option to either append (default)
    or rewrite by passing in Boolean value to append_to_json.
    """
    if verbose:
        logging.info(f"Saving data to file: {filename}")
    file_ext = os.path.splitext(filename)[1]
    if file_ext in [".pkl", ".pickle"]:
        with g_pathmgr.open(filename, "wb") as fopen:
            pickle.dump(data, fopen, pickle.HIGHEST_PROTOCOL)
    elif file_ext == ".npy":
        with g_pathmgr.open(filename, "wb") as fopen:
            np.save(fopen, data)
    elif file_ext == ".json":
        if append_to_json:
            with g_pathmgr.open(filename, "a") as fopen:
                fopen.write(json.dumps(data, sort_keys=True) + "\n")
                fopen.flush()
        else:
            with g_pathmgr.open(filename, "w") as fopen:
                fopen.write(json.dumps(data, sort_keys=True) + "\n")
                fopen.flush()
    elif file_ext == ".yaml":
        with g_pathmgr.open(filename, "w") as fopen:
            dump = yaml.dump(data)
            fopen.write(dump)
            fopen.flush()
    else:
        raise Exception(f"Saving {file_ext} is not supported yet")

    if verbose:
        logging.info(f"Saved data to file: {filename}")


def load_file(filename, mmap_mode=None, verbose=True, allow_pickle=False):
    """
    Common i/o utility to handle loading data from various file formats.
    Supported:
        .pkl, .pickle, .npy, .json
    For the npy files, we support reading the files in mmap_mode.
    If the mmap_mode of reading is not successful, we load data without the
    mmap_mode.
    """
    if verbose:
        logging.info(f"Loading data from file: {filename}")

    file_ext = os.path.splitext(filename)[1]
    if file_ext == ".txt":
        with g_pathmgr.open(filename, "r") as fopen:
            data = fopen.readlines()
    elif file_ext in [".pkl", ".pickle"]:
        with g_pathmgr.open(filename, "rb") as fopen:
            data = pickle.load(fopen, encoding="latin1")
    elif file_ext == ".npy":
        if mmap_mode:
            try:
                with g_pathmgr.open(filename, "rb") as fopen:
                    data = np.load(
                        fopen,
                        allow_pickle=allow_pickle,
                        encoding="latin1",
                        mmap_mode=mmap_mode,
                    )
            except ValueError as e:
                logging.info(f"Could not mmap {filename}: {e}. Trying without g_pathmgr")
                data = np.load(
                    filename,
                    allow_pickle=allow_pickle,
                    encoding="latin1",
                    mmap_mode=mmap_mode,
                )
                logging.info("Successfully loaded without g_pathmgr")
            except Exception:
                logging.info("Could not mmap without g_pathmgr. Trying without mmap")
                with g_pathmgr.open(filename, "rb") as fopen:
                    data = np.load(fopen, allow_pickle=allow_pickle, encoding="latin1")
        else:
            with g_pathmgr.open(filename, "rb") as fopen:
                data = np.load(fopen, allow_pickle=allow_pickle, encoding="latin1")
    elif file_ext == ".json":
        with g_pathmgr.open(filename, "r") as fopen:
            data = json.load(fopen)
    elif file_ext == ".yaml":
        with g_pathmgr.open(filename, "r") as fopen:
            data = yaml.load(fopen, Loader=yaml.FullLoader)
    elif file_ext == ".csv":
        with g_pathmgr.open(filename, "r") as fopen:
            data = pd.read_csv(fopen)
    else:
        raise Exception(f"Reading from {file_ext} is not supported yet")
    return data


def abspath(resource_path: str):
    """
    Make a path absolute, but take into account prefixes like
    "http://" or "manifold://"
    """
    regex = re.compile(r"^\w+://")
    if regex.match(resource_path) is None:
        return os.path.abspath(resource_path)
    else:
        return resource_path


def makedir(dir_path):
    """
    Create the directory if it does not exist.
    """
    is_success = False
    try:
        if not g_pathmgr.exists(dir_path):
            g_pathmgr.mkdirs(dir_path)
        is_success = True
    except BaseException:
        logging.info(f"Error creating directory: {dir_path}")
    return is_success


def is_url(input_url):
    """
    Check if an input string is a url. look for http(s):// and ignoring the case
    """
    is_url = re.match(r"^(?:http)s?://", input_url, re.IGNORECASE) is not None
    return is_url


def cleanup_dir(dir):
    """
    Utility for deleting a directory. Useful for cleaning the storage space
    that contains various training artifacts like checkpoints, data etc.
    """
    if os.path.exists(dir):
        logging.info(f"Deleting directory: {dir}")
        shutil.rmtree(dir)
    logging.info(f"Deleted contents of directory: {dir}")


def get_file_size(filename):
    """
    Given a file, get the size of file in MB
    """
    size_in_mb = os.path.getsize(filename) / float(1024**2)
    return size_in_mb


def scale_img(img):
    return 2 * img - 1


def parse_loc_answer(answer):
    """
    "<loc1><loc2><loc3>" -> [1, 2, 3]
    """
    answer = answer.replace("<", "").replace(">", "")
    answer = answer.split("loc")
    answer = [x for x in answer if x != ""]
    answer = [int(x) for x in answer]
    return answer


def psnr(img1, img2):
    """
    Compute Peak Signal to Noise Ratio (the higher the better).
    img1 and img2 have values in [0, 1]
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    if img1.shape != img2.shape:
        # (244, 244, 3) -> (3, 244, 244)
        img1 = np.transpose(img1, (2, 0, 1))
    img1 = img1.astype(np.float64) * 255.0
    img2 = img2.astype(np.float64) * 255.0
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def apply_with_stopping_condition(module, apply_fn, apply_condition=None, stopping_condition=None, **other_args):
    if stopping_condition(module):
        return
    if apply_condition(module):
        apply_fn(module, **other_args)
    for child in module.children():
        apply_with_stopping_condition(
            child, apply_fn, apply_condition=apply_condition, stopping_condition=stopping_condition, **other_args
        )


def apply_activation_checkpointing(model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=lambda _: True):
    """
    Applies :func:`checkpoint_wrapper` to modules within `model` based on a user-defined
    configuration. For each module within `model`, the `check_fn` is used to decide
    whether `module` should be wrapped with :func:`checkpoint_wrapper` or not.

    Note::
        This function modifies `model` in place and replaces appropriate layers with
        their checkpoint-wrapped modules.
    Note::
        This function will not wrap the overall root module. If this is needed, please directly use
        :class:`CheckpointWrapper`.
    Usage::
        model = nn.Sequential(
            nn.Linear(10, 10), nn.Linear(10, 10), nn.Linear(10, 10)
        )
        check_fn = lambda l: isinstance(l, nn.Linear)
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn)
    Args:
        model (nn.Module):
            The model whose submodules should be wrapped with activation checkpointing.
        checkpoint_wrapper_fn (Optional[Callable[nn.Module]])
            A ``Callable`` which will wrap modules
        check_fn (Optional[Callable[nn.Module, nn.Module]])
            A lambda function which will be passed each child submoule of ``model`` and returns
            ``True`` or ``False`` depending on whether the submodule should be wrapped.
    Returns: None (`model` is modified inplace)
    """
    # TODO: Importing inside function to avoid circular import issue between FSDP and
    # checkpoint_wrapper. This can be resolved once wrap() APIs are decoupled from FSDP code.
    from torch.distributed.fsdp.wrap import _recursive_wrap, lambda_auto_wrap_policy

    return _recursive_wrap(
        module=model,
        auto_wrap_policy=partial(lambda_auto_wrap_policy, lambda_fn=check_fn),
        wrapper_cls=checkpoint_wrapper_fn,
        ignored_modules=set(),
        ignored_params=set(),
        only_wrap_children=True,
    )
