"""A thin wrapper for storage cache using datumaro arrow format exporter."""
# copyright (c) 2023 intel corporation
# spdx-license-identifier: apache-2.0
#

import hashlib
import os
import shutil
import stat
from typing import List, Optional

from datumaro.components.dataset import Dataset as DatumDataset
from datumaro.components.progress_reporting import SimpleProgressReporter

from otx.core.file import OTX_CACHE

DATASET_CACHE = os.path.join(OTX_CACHE, "dataset")


def arrow_cache_helper(
    dataset: DatumDataset,
    scheme: str,
    num_workers: int = 0,
    cache_dir: str = DATASET_CACHE,
    force: bool = False,
) -> List[str]:
    """A helper for dumping Datumaro arrow format.

    Args:
        dataset: Datumaro dataset to export in apache arrow.
        scheme: Datumaro apache arrow image encoding scheme.
        num_workers: The number of workers to build arrow format.
        cache_dir: The directory to save.
        force: If true, rebuild arrow even if cache is hit.
    """

    def get_hash(dataset, scheme):
        source_path = dataset.data_path
        _hash = hashlib.sha256()
        _hash.update(f"{source_path}".encode("utf-8"))
        _hash.update(f"{len(dataset)}".encode("utf-8"))
        _hash.update(f"{scheme}".encode("utf-8"))
        if source_path:
            _hash.update(str(os.stat(source_path)[stat.ST_MTIME]).encode("utf-8"))
            if os.path.isdir(source_path):
                for root, dirs, files in os.walk(source_path):
                    for file in files:
                        file = os.path.join(root, file)
                        _hash.update(str(os.stat(file)[stat.ST_MTIME]).encode("utf-8"))
                    for _dir in dirs:
                        _dir = os.path.join(root, _dir)
                        _hash.update(str(os.stat(_dir)[stat.ST_MTIME]).encode("utf-8"))
        return _hash.hexdigest()

    def get_file_hash(file):
        _hash = hashlib.sha256()
        _hash.update(str(file).encode("utf-8"))
        _hash.update(str(os.stat(file)[stat.ST_MTIME]).encode("utf-8"))
        return _hash.hexdigest()

    cache_dir = os.path.join(cache_dir, get_hash(dataset, scheme))
    if os.path.exists(cache_dir) and force:
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    cache_paths = []
    cache_hit = []
    for cache_path in os.listdir(cache_dir):
        if not cache_path.endswith(".arrow"):
            continue
        cache_hit.append(False)
        cache_path = os.path.join(cache_dir, cache_path)
        cache_paths.append(cache_path)
        hash_path = f"{cache_path}.hash"
        if os.path.exists(cache_path) and os.path.exists(hash_path):
            with open(hash_path, "r", encoding="utf-8") as f:
                if get_file_hash(cache_path) == f.read():
                    cache_hit[-1] = True

    if cache_hit and all(cache_hit):
        return cache_paths

    dataset.export(
        cache_dir,
        "arrow",
        save_media=True,
        image_ext=scheme,
        num_workers=num_workers,
        progress_reporter=SimpleProgressReporter(0, 10),
    )

    cache_paths = []
    for cache_path in os.listdir(cache_dir):
        if not cache_path.endswith(".arrow"):
            continue
        cache_path = os.path.join(cache_dir, cache_path)
        cache_paths.append(cache_path)
        hash_path = f"{cache_path}.hash"
        with open(hash_path, "w", encoding="utf-8") as f:
            f.write(get_file_hash(cache_path))

    return cache_paths


def init_arrow_cache(dataset: DatumDataset, scheme: Optional[str] = None, **kwargs) -> DatumDataset:
    """Init arrow format cache from Datumaro.

    Args:
        dataset: Datumaro dataset
        scheme: Datumaro apache arrow image encoding scheme.
        kwargs: kwargs passed to 'arrow_cache_helper'
    """
    if scheme is None or scheme == "NONE":
        return dataset
    cache_paths = arrow_cache_helper(dataset, scheme, **kwargs)
    dataset = DatumDataset.import_from(os.path.dirname(cache_paths[0]), "arrow")
    return dataset
