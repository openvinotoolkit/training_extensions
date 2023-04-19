# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


from copy import deepcopy
import os
import stat
import tempfile
import time

import numpy as np
import pytest
from datumaro.components.annotation import Label, Polygon, Bbox, Mask
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image

from otx.core.data.caching.storage_cache import init_arrow_cache


@pytest.fixture
def fxt_datumaro_dataset():
    items = []
    for i in range(64):
        media = Image.from_numpy(data=np.random.randint(0, 255, (5, 5, 3)), ext=".png")

        items.append(
            DatasetItem(
                id=i,
                subset="test",
                media=media,
                annotations=[
                    # annotations used in OTX
                    Label(np.random.randint(0, 3)),
                    Bbox(*np.random.randint(0, 5, 4)),
                    Polygon(Bbox(*np.random.randint(0, 5, 4)).as_polygon()),
                    Mask(np.random.randint(0, 2, (5, 5))),
                ],
            )
        )

    source_dataset = Dataset.from_iterable(
        items,
        categories=["label"],
        media_type=Image,
    )

    return source_dataset


def compare_dataset(source_dataset, target_dataset, compare_media=True):
    properties = ["id", "subset", "annotations", "attributes"]
    if compare_media:
        properties.append("media")
    for item_s, item_t in zip(source_dataset, target_dataset):
        for property in properties:
            assert getattr(item_s, property) == getattr(item_t, property)


class TestStorageCache:
    @pytest.mark.parametrize(
        ["scheme", "compare_media"],
        [
            pytest.param(
                "NONE",
                True,
                id="test_none_scheme",
            ),
            pytest.param(
                "AS-IS",
                True,
                id="test_as_is_scheme",
            ),
            pytest.param(
                "PNG",
                True,
                id="test_png_scheme",
            ),
            pytest.param(
                "TIFF",
                True,
                id="test_tiff_scheme",
            ),
            pytest.param(
                "JPEG/95",
                False,
                id="test_jpeg_95_scheme",
            ),
            pytest.param(
                "JPEG/75",
                False,
                id="test_jpeg_75_scheme",
            ),
        ],
    )
    def test_is_identical(self, scheme, fxt_datumaro_dataset, compare_media):
        with tempfile.TemporaryDirectory() as tempdir:
            source_dataset = fxt_datumaro_dataset
            cached_dataset = init_arrow_cache(source_dataset, scheme=scheme, cache_dir=tempdir)
            compare_dataset(source_dataset, cached_dataset, compare_media)

    def test_cache_hit(self, fxt_datumaro_dataset):
        with tempfile.TemporaryDirectory() as tempdir:
            cached_dataset = init_arrow_cache(deepcopy(fxt_datumaro_dataset), scheme="AS-IS", cache_dir=tempdir)

            mapping = {}
            for file in os.listdir(cached_dataset.data_path):
                mapping[file] = os.stat(os.path.join(cached_dataset.data_path, file))[stat.ST_MTIME]

            cached_dataset = init_arrow_cache(deepcopy(fxt_datumaro_dataset), scheme="AS-IS", cache_dir=tempdir)

            for file in os.listdir(cached_dataset.data_path):
                assert mapping[file] == os.stat(os.path.join(cached_dataset.data_path, file))[stat.ST_MTIME]

    def test_no_cache_hit(self, fxt_datumaro_dataset):
        with tempfile.TemporaryDirectory() as tempdir:
            cached_dataset = init_arrow_cache(deepcopy(fxt_datumaro_dataset), scheme="AS-IS", cache_dir=tempdir)

            mapping = {}
            for file in os.listdir(cached_dataset.data_path):
                mapping[file] = os.stat(os.path.join(cached_dataset.data_path, file))[stat.ST_MTIME]

            # sleep 1 second to invalidate cache
            time.sleep(1)

            cached_dataset = init_arrow_cache(
                deepcopy(fxt_datumaro_dataset), scheme="AS-IS", cache_dir=tempdir, force=True
            )

            for file in os.listdir(cached_dataset.data_path):
                assert mapping[file] != os.stat(os.path.join(cached_dataset.data_path, file))[stat.ST_MTIME]
