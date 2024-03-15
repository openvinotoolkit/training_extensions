# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest

from otx.algorithms.segmentation.adapters.mmseg.datasets.pipelines import (
    LoadAnnotationFromOTXDataset,
    LoadResizeDataFromOTXDataset,
)
from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.image import Image
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle
from otx.core.data.caching import MemCacheHandlerSingleton

from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.detection.test_helpers import generate_det_dataset

from typing import Iterator, List, Optional, Sequence, Tuple


def label_entity(name="test label") -> LabelEntity:
    return LabelEntity(name=name, domain=Domain.SEGMENTATION)


def dataset_item() -> DatasetItemEntity:
    image: Image = Image(data=np.random.randint(low=0, high=255, size=(10, 16, 3)).astype(np.uint8))
    annotation: Annotation = Annotation(shape=Rectangle.generate_full_box(), labels=[ScoredLabel(label_entity())])
    annotation_scene: AnnotationSceneEntity = AnnotationSceneEntity(
        annotations=[annotation], kind=AnnotationSceneKind.ANNOTATION
    )
    return DatasetItemEntity(media=image, annotation_scene=annotation_scene)


class TestLoadAnnotationFromOTXDataset:
    @pytest.fixture(autouse=True)
    def setUp(self) -> None:

        self.dataset_item: DatasetItemEntity = dataset_item()
        self.results: dict = {
            "dataset_item": self.dataset_item,
            "ann_info": {"labels": [label_entity("class_1")]},
            "seg_fields": [],
        }
        self.pipeline: LoadAnnotationFromOTXDataset = LoadAnnotationFromOTXDataset()

    @e2e_pytest_unit
    def test_call(self) -> None:
        loaded_annotations: dict = self.pipeline(self.results)
        assert "gt_semantic_seg" in loaded_annotations


@e2e_pytest_unit
@pytest.mark.parametrize("mode", ["singleprocessing", "multiprocessing"])
def test_load_resize_data_from_otx_dataset_call(mocker, mode):
    """Test LoadResizeDataFromOTXDataset."""
    item: DatasetItemEntity = dataset_item()
    MemCacheHandlerSingleton.create(mode, item.numpy.size * 2)
    op = LoadResizeDataFromOTXDataset(
        use_otx_adapter=True,
        load_ann_cfg=dict(
            type="LoadAnnotationFromOTXDataset",
            use_otx_adapter=True,
        ),
        resize_cfg=dict(
            type="Resize",
            img_scale=(8, 5),  # (w, h)
        ),  # 10x16 -> 5x8
    )
    src_dict = dict(
        dataset_item=item,
        width=item.width,
        height=item.height,
        index=0,
        ann_info=dict(labels=[label_entity()]),
        seg_fields=[],
    )
    dst_dict = op(src_dict)
    assert dst_dict["ori_shape"][0] == 10
    assert dst_dict["img_shape"][0] == 5  # height
    assert dst_dict["img_shape"][1] == 8  # width
    assert dst_dict["img"].shape == dst_dict["img_shape"]
    assert dst_dict["img"].shape[:2] == dst_dict["gt_semantic_seg"].shape[:2]
    assert "gt_semantic_seg" not in src_dict  # src_dict not affected
    op._load_img = mocker.MagicMock()
    dst_dict_from_cache = op(src_dict)
    assert op._load_img.call_count == 0  # _load_img() should not be called
    assert np.array_equal(dst_dict["img"], dst_dict_from_cache["img"])
    assert np.array_equal(dst_dict["gt_semantic_seg"], dst_dict_from_cache["gt_semantic_seg"])
