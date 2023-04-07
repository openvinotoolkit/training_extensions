"""A thin wrapper for storage cache using datumaro arrow format exporter."""
# copyright (c) 2023 intel corporation
# spdx-license-identifier: apache-2.0
#

import hashlib
import os
import stat
from typing import List, Optional

from datumaro.components.dataset import Dataset as DatumDataset

from otx.core.file import OTX_CACHE

DATASET_CACHE = os.path.join(OTX_CACHE, "dataset")


def arrow_cache_helper(
    dataset: DatumDataset,
    scheme: str,
    force: bool = False,
) -> List[str]:
    """A helper for dumping Datumaro arrow format."""

    def get_hash(source_dir, scheme):
        _hash = hashlib.sha256()
        _hash.update(f"{source_dir}".encode("utf-8"))
        _hash.update(f"{scheme}".encode("utf-8"))
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                _hash.update(str(os.stat(os.path.join(root, file))[stat.ST_MTIME]).encode("utf-8"))
            for _dir in dirs:
                _hash.update(str(os.stat(os.path.join(root, _dir))[stat.ST_MTIME]).encode("utf-8"))
        return _hash.hexdigest()

    def get_file_hash(file):
        _hash = hashlib.sha256()
        _hash.update(str(file).encode("utf-8"))
        _hash.update(str(os.stat(file)[stat.ST_MTIME]).encode("utf-8"))
        return _hash.hexdigest()

    cache_dir = get_hash(dataset.data_path, scheme)
    cache_dir = os.path.join(DATASET_CACHE, cache_dir)
    cache_paths = []
    os.makedirs(cache_dir, exist_ok=True)

    cache_hit = [False for _ in dataset.subsets().keys()]
    for i, subset in enumerate(dataset.subsets().keys()):
        cache_path = os.path.join(cache_dir, f"{subset}.arrow")
        cache_paths.append(cache_path)
        hash_path = f"{cache_path}.hash"
        if os.path.exists(cache_path) and os.path.exists(hash_path) and not force:
            with open(hash_path, "r", encoding="utf-8") as f:
                if get_file_hash(cache_path) == f.read():
                    cache_hit[i] = True

    if all(cache_hit):
        return cache_paths

    dataset.export(cache_dir, "arrow", save_media=True, image_ext=scheme)

    for cache_path in cache_paths:
        hash_path = f"{cache_path}.hash"
        with open(hash_path, "w", encoding="utf-8") as f:
            f.write(get_file_hash(cache_path))

    return cache_paths


def init_arrow_cache(dataset: DatumDataset, scheme: Optional[str] = None) -> DatumDataset:
    """Init arrow format cache from Datumaro."""
    if scheme is None or scheme == "NONE":
        return dataset
    cache_paths = arrow_cache_helper(dataset, scheme)
    datasets = []
    for cache_path in cache_paths:
        datasets.append(DatumDataset.import_from(cache_path, "arrow"))
    dataset = DatumDataset.from_extractors(*datasets)
    if len(cache_paths) == 1:
        dataset._source_path = cache_paths[0]  # pylint: disable=protected-access
    else:
        dataset._source_path = os.path.dirname(cache_paths[0])  # pylint: disable=protected-access

    return dataset
