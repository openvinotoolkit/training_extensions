# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from datumaro.components.annotation import AnnotationType, Bbox, Label, LabelCategories, Mask, Polygon
from datumaro.components.dataset import Dataset as DmDataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from otx.core.data.dataset.action_classification import (
    ActionClsDataEntity,
    OTXActionClsDataset,
)
from otx.core.data.dataset.anomaly import (
    AnomalyClassificationDataItem,
    AnomalyDataset,
    AnomalyDetectionDataItem,
    AnomalySegmentationDataItem,
)
from otx.core.data.dataset.classification import (
    HlabelClsDataEntity,
    HLabelInfo,
    MulticlassClsDataEntity,
    MultilabelClsDataEntity,
    OTXHlabelClsDataset,
    OTXMulticlassClsDataset,
    OTXMultilabelClsDataset,
)
from otx.core.data.dataset.detection import (
    DetDataEntity,
    OTXDetectionDataset,
)
from otx.core.data.dataset.instance_segmentation import (
    InstanceSegDataEntity,
    OTXInstanceSegDataset,
)
from otx.core.data.dataset.segmentation import (
    OTXSegmentationDataset,
    SegDataEntity,
)
from otx.core.data.mem_cache import MemCacheHandlerSingleton
from otx.core.types.task import OTXTaskType

if TYPE_CHECKING:
    from otx.core.data.dataset.base import OTXDataset, T_OTXDataEntity
    from otx.core.data.mem_cache import MemCacheHandlerBase
    from pytest_mock import MockerFixture

_LABEL_NAMES = ["Non-Rigid", "Rigid", "Rectangle", "Triangle", "Circle", "Lion", "Panda"]


@pytest.fixture()
def fxt_mem_cache_handler(monkeypatch) -> MemCacheHandlerBase:
    monkeypatch.setattr(MemCacheHandlerSingleton, "check_system_memory", lambda *_: True)
    handler = MemCacheHandlerSingleton.create(mode="singleprocessing", mem_size=1024 * 1024)
    yield handler
    MemCacheHandlerSingleton.delete()


@pytest.fixture(params=["bytes", "file"])
def fxt_dm_item(request, tmpdir) -> DatasetItem:
    np_img = np.zeros(shape=(10, 10, 3), dtype=np.uint8)
    np_img[:, :, 0] = 0  # Set 0 for B channel
    np_img[:, :, 1] = 1  # Set 1 for G channel
    np_img[:, :, 2] = 2  # Set 2 for R channel

    if request.param == "bytes":
        _, np_bytes = cv2.imencode(".png", np_img)
        media = Image.from_bytes(np_bytes.tobytes())
        media.path = ""
    elif request.param == "file":
        fname = str(uuid.uuid4())
        fpath = str(Path(tmpdir) / f"{fname}.png")
        cv2.imwrite(fpath, np_img)
        media = Image.from_file(fpath)
    else:
        raise ValueError(request.param)

    return DatasetItem(
        id="item",
        subset="train",
        media=media,
        annotations=[
            Label(label=0),
            Bbox(x=200, y=200, w=1, h=1, label=0),
            Mask(label=0, image=np.eye(10, dtype=np.uint8)),
            Polygon(points=[399.0, 570.0, 397.0, 572.0, 397.0, 573.0, 394.0, 576.0], label=0),
        ],
    )


@pytest.fixture(params=["bytes", "file"])
def fxt_dm_item_bbox_only(request, tmpdir) -> DatasetItem:
    np_img = np.zeros(shape=(10, 10, 3), dtype=np.uint8)
    np_img[:, :, 0] = 0  # Set 0 for B channel
    np_img[:, :, 1] = 1  # Set 1 for G channel
    np_img[:, :, 2] = 2  # Set 2 for R channel

    if request.param == "bytes":
        _, np_bytes = cv2.imencode(".png", np_img)
        media = Image.from_bytes(np_bytes.tobytes())
        media.path = ""
    elif request.param == "file":
        fname = str(uuid.uuid4())
        fpath = str(Path(tmpdir) / f"{fname}.png")
        cv2.imwrite(fpath, np_img)
        media = Image.from_file(fpath)
    else:
        raise ValueError(request.param)

    return DatasetItem(
        id="item",
        subset="train",
        media=media,
        annotations=[
            Bbox(x=0, y=0, w=1, h=1, label=0),
            Bbox(x=1, y=0, w=1, h=1, label=0),
            Bbox(x=1, y=1, w=1, h=1, label=0),
        ],
    )


@pytest.fixture()
def fxt_mock_dm_subset(mocker: MockerFixture, fxt_dm_item: DatasetItem) -> MagicMock:
    mock_dm_subset = mocker.MagicMock(spec=DmDataset)
    mock_dm_subset.__getitem__.return_value = fxt_dm_item
    mock_dm_subset.__len__.return_value = 1
    mock_dm_subset.categories().__getitem__.return_value = LabelCategories.from_iterable(_LABEL_NAMES)
    mock_dm_subset.ann_types.return_value = [
        AnnotationType.label,
        AnnotationType.bbox,
        AnnotationType.mask,
        AnnotationType.polygon,
    ]
    return mock_dm_subset


@pytest.fixture()
def fxt_mock_det_dm_subset(mocker: MockerFixture, fxt_dm_item_bbox_only: DatasetItem) -> MagicMock:
    mock_dm_subset = mocker.MagicMock(spec=DmDataset)
    mock_dm_subset.__getitem__.return_value = fxt_dm_item_bbox_only
    mock_dm_subset.__len__.return_value = 1
    mock_dm_subset.categories().__getitem__.return_value = LabelCategories.from_iterable(_LABEL_NAMES)
    mock_dm_subset.ann_types.return_value = [AnnotationType.bbox]
    return mock_dm_subset


@pytest.fixture(
    params=[
        (OTXHlabelClsDataset, HlabelClsDataEntity, {}),
        (OTXMultilabelClsDataset, MultilabelClsDataEntity, {}),
        (OTXMulticlassClsDataset, MulticlassClsDataEntity, {}),
        (OTXDetectionDataset, DetDataEntity, {}),
        (OTXInstanceSegDataset, InstanceSegDataEntity, {"include_polygons": True}),
        (OTXSegmentationDataset, SegDataEntity, {}),
        (OTXActionClsDataset, ActionClsDataEntity, {}),
        (AnomalyDataset, AnomalyClassificationDataItem, {"task_type": OTXTaskType.ANOMALY_CLASSIFICATION}),
        (AnomalyDataset, AnomalyDetectionDataItem, {"task_type": OTXTaskType.ANOMALY_DETECTION}),
        (AnomalyDataset, AnomalySegmentationDataItem, {"task_type": OTXTaskType.ANOMALY_SEGMENTATION}),
    ],
    ids=[
        "hlabel_cls",
        "multi_label_cls",
        "multi_class_cls",
        "detection",
        "instance_seg",
        "semantic_seg",
        "action_cls",
        "anomaly_cls",
        "anomaly_det",
        "anomaly_seg",
    ],
)
def fxt_dataset_and_data_entity_cls(
    request: pytest.FixtureRequest,
) -> tuple[OTXDataset, T_OTXDataEntity]:
    return request.param


@pytest.fixture()
def fxt_mock_hlabelinfo():
    mock_dict = MagicMock()
    mock_dict.__getitem__.return_value = (0, 0)
    return HLabelInfo(
        label_names=_LABEL_NAMES,
        label_groups=[["Non-Rigid", "Rigid"], ["Rectangle", "Triangle"], ["Circle"], ["Lion"], ["Panda"]],
        num_multiclass_heads=2,
        num_multilabel_classes=3,
        head_idx_to_logits_range={"0": (0, 2), "1": (2, 4)},
        num_single_label_classes=4,
        class_to_group_idx=mock_dict,
        all_groups=[["Non-Rigid", "Rigid"], ["Rectangle", "Triangle"], ["Circle"], ["Lion"], ["Panda"]],
        label_to_idx={
            "Rigid": 0,
            "Rectangle": 1,
            "Triangle": 2,
            "Non-Rigid": 3,
            "Circle": 4,
            "Lion": 5,
            "Panda": 6,
        },
        label_tree_edges=[
            ["Rectangle", "Rigid"],
            ["Triangle", "Rigid"],
            ["Circle", "Non-Rigid"],
        ],
        empty_multiclass_head_indices=[],
    )
