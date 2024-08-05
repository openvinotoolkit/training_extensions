# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch
from datumaro import Polygon
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.classification import (
    HlabelClsBatchDataEntity,
    HlabelClsBatchPredEntity,
    HlabelClsDataEntity,
    MulticlassClsBatchDataEntity,
    MulticlassClsBatchPredEntity,
    MulticlassClsDataEntity,
    MultilabelClsBatchDataEntity,
    MultilabelClsBatchPredEntity,
    MultilabelClsDataEntity,
)
from otx.core.data.entity.detection import DetBatchDataEntity, DetBatchPredEntity, DetDataEntity
from otx.core.data.entity.instance_segmentation import (
    InstanceSegBatchDataEntity,
    InstanceSegBatchPredEntity,
    InstanceSegDataEntity,
)
from otx.core.data.entity.segmentation import SegBatchDataEntity, SegBatchPredEntity, SegDataEntity
from otx.core.data.mem_cache import MemCacheHandlerSingleton
from otx.core.types.label import HLabelInfo, LabelInfo, NullLabelInfo, SegLabelInfo
from otx.core.types.task import OTXTaskType
from torch import LongTensor
from torchvision import tv_tensors
from torchvision.tv_tensors import Image, Mask

from tests.utils import ExportCase2Test


def pytest_addoption(parser: pytest.Parser):
    """Add custom options for perf tests."""
    parser.addoption(
        "--model-category",
        action="store",
        default="all",
        choices=("speed", "balance", "accuracy", "default", "other", "all"),
        help="Choose speed|balcence|accuracy|default|other|all. Defaults to all.",
    )
    parser.addoption(
        "--data-group",
        action="store",
        default="all",
        choices=("small", "medium", "large", "all"),
        help="Choose small|medium|large|all. Defaults to all.",
    )
    parser.addoption(
        "--num-repeat",
        action="store",
        default=0,
        help="Overrides default per-data-group number of repeat setting. "
        "Random seeds are set to 0 ~ num_repeat-1 for the trials. "
        "Defaults to 0 (small=3, medium=3, large=1).",
    )
    parser.addoption(
        "--num-epoch",
        action="store",
        default=0,
        help="Overrides default per-model number of epoch setting. "
        "Defaults to 0 (per-model epoch & early-stopping).",
    )
    parser.addoption(
        "--eval-upto",
        action="store",
        default="train",
        choices=("train", "export", "optimize"),
        help="Choose train|export|optimize. Defaults to train.",
    )
    parser.addoption(
        "--data-root",
        action="store",
        default="data",
        help="Dataset root directory.",
    )
    parser.addoption(
        "--output-root",
        action="store",
        help="Output root directory. Defaults to temp directory.",
    )
    parser.addoption(
        "--summary-file",
        action="store",
        help="Path to output summary file. Defaults to {output-root}/benchmark-summary.csv",
    )
    parser.addoption(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print OTX commands without execution.",
    )
    parser.addoption(
        "--deterministic",
        choices=["true", "false", "warn"],
        default=None,
        help="Turn on deterministic training (true/false/warn).",
    )
    parser.addoption(
        "--user-name",
        type=str,
        default="anonymous",
        help='Sign-off the user name who launched the regression tests this time, e.g., `--user-name "John Doe"`.',
    )
    parser.addoption(
        "--mlflow-tracking-uri",
        type=str,
        help="URI for MLFlow Tracking server to store the regression test results.",
    )
    parser.addoption(
        "--otx-ref",
        type=str,
        default="__CURRENT_BRANCH_COMMIT__",
        help="Target OTX ref (tag / branch name / commit hash) on main repo to test. Defaults to the current branch. "
        "`pip install otx[full]@https://github.com/openvinotoolkit/training_extensions.git@{otx_ref}` will be executed before run, "
        "and reverted after run. Works only for v2.x assuming CLI compatibility.",
    )
    parser.addoption(
        "--resume-from",
        type=str,
        help="Previous performance test directory which contains execution results. "
        "If training was already done in previous performance test, training is skipped and refer previous result.",
    )
    parser.addoption(
        "--test-only",
        action="store",
        choices=("all", "train", "export", "optimize"),
        help="Execute test only when resume argument is given. If necessary files are not found in resume directory, "
        "necessary operations can be executed. Choose all|train|export|optimize.",
    )
    parser.addoption(
        "--open-subprocess",
        action="store_true",
        help="Open subprocess for each CLI test case. "
        "This option can be used for easy memory management "
        "while running consecutive multiple tests (default: false).",
    )
    parser.addoption(
        "--task",
        action="store",
        default="all",
        type=str,
        help="Task type of OTX to use test.",
    )
    parser.addoption(
        "--device",
        action="store",
        default="gpu",
        type=str,
        help="Which device to use.",
    )


@pytest.fixture(scope="session")
def fxt_multi_class_cls_data_entity() -> (
    tuple[MulticlassClsDataEntity, MulticlassClsBatchDataEntity, MulticlassClsBatchDataEntity]
):
    img_size = (64, 64)
    fake_image = torch.zeros(size=(3, *img_size), dtype=torch.uint8).numpy()
    fake_image_info = ImageInfo(img_idx=0, img_shape=img_size, ori_shape=img_size)
    fake_labels = LongTensor([0])
    fake_score = torch.Tensor([0.6])
    # define data entity
    single_data_entity = MulticlassClsDataEntity(fake_image, fake_image_info, fake_labels)
    batch_data_entity = MulticlassClsBatchDataEntity(
        batch_size=1,
        images=[Image(data=torch.from_numpy(fake_image))],
        imgs_info=[fake_image_info],
        labels=[fake_labels],
    )
    batch_pred_data_entity = MulticlassClsBatchPredEntity(
        batch_size=1,
        images=[Image(data=torch.from_numpy(fake_image))],
        imgs_info=[fake_image_info],
        labels=[fake_labels],
        scores=[fake_score],
    )

    return single_data_entity, batch_pred_data_entity, batch_data_entity


@pytest.fixture(scope="session")
def fxt_multi_label_cls_data_entity() -> (
    tuple[MultilabelClsDataEntity, MultilabelClsBatchDataEntity, MultilabelClsBatchDataEntity]
):
    img_size = (64, 64)
    fake_image = torch.zeros(size=(3, *img_size), dtype=torch.uint8).numpy()
    fake_image_info = ImageInfo(img_idx=0, img_shape=img_size, ori_shape=img_size)
    fake_labels = LongTensor([0])
    fake_score = torch.Tensor([0.6])
    # define data entity
    single_data_entity = MultilabelClsDataEntity(fake_image, fake_image_info, fake_labels)
    batch_data_entity = MultilabelClsBatchDataEntity(
        batch_size=1,
        images=[Image(data=torch.from_numpy(fake_image))],
        imgs_info=[fake_image_info],
        labels=[fake_labels],
    )
    batch_pred_data_entity = MultilabelClsBatchPredEntity(
        batch_size=1,
        images=[Image(data=torch.from_numpy(fake_image))],
        imgs_info=[fake_image_info],
        labels=[fake_labels],
        scores=[fake_score],
    )

    return single_data_entity, batch_pred_data_entity, batch_data_entity


@pytest.fixture(scope="session")
def fxt_h_label_cls_data_entity() -> tuple[HlabelClsDataEntity, HlabelClsBatchDataEntity, HlabelClsBatchPredEntity]:
    img_size = (64, 64)
    fake_image = torch.zeros(size=(3, *img_size), dtype=torch.uint8).numpy()
    fake_image_info = ImageInfo(img_idx=0, img_shape=img_size, ori_shape=img_size)
    fake_labels = LongTensor([0])
    fake_score = torch.Tensor([0.6])
    # define data entity
    single_data_entity = HlabelClsDataEntity(fake_image, fake_image_info, fake_labels)
    batch_data_entity = HlabelClsBatchDataEntity(
        batch_size=1,
        images=[Image(data=torch.from_numpy(fake_image))],
        imgs_info=[fake_image_info],
        labels=[fake_labels],
    )
    batch_pred_data_entity = HlabelClsBatchPredEntity(
        batch_size=1,
        images=[Image(data=torch.from_numpy(fake_image))],
        imgs_info=[fake_image_info],
        labels=[fake_labels],
        scores=[fake_score],
    )

    return single_data_entity, batch_pred_data_entity, batch_data_entity


@pytest.fixture(scope="session")
def fxt_det_data_entity() -> tuple[tuple, DetDataEntity, DetBatchDataEntity]:
    img_size = (64, 64)
    fake_image = torch.zeros(size=(3, *img_size), dtype=torch.float32).numpy()
    fake_image_info = ImageInfo(img_idx=0, img_shape=img_size, ori_shape=img_size)
    fake_bboxes = tv_tensors.BoundingBoxes(data=torch.Tensor([0, 0, 5, 5]), format="xyxy", canvas_size=(10, 10))
    fake_labels = LongTensor([1])
    # define data entity
    single_data_entity = DetDataEntity(fake_image, fake_image_info, fake_bboxes, fake_labels)
    batch_data_entity = DetBatchDataEntity(
        batch_size=1,
        images=[Image(data=torch.from_numpy(fake_image))],
        imgs_info=[fake_image_info],
        bboxes=[fake_bboxes],
        labels=[fake_labels],
    )
    batch_pred_data_entity = DetBatchPredEntity(
        batch_size=1,
        images=[Image(data=torch.from_numpy(fake_image))],
        imgs_info=[fake_image_info],
        bboxes=[fake_bboxes],
        labels=[fake_labels],
        scores=[],
    )

    return single_data_entity, batch_pred_data_entity, batch_data_entity


@pytest.fixture(scope="session")
def fxt_inst_seg_data_entity() -> tuple[tuple, InstanceSegDataEntity, InstanceSegBatchDataEntity]:
    img_size = (64, 64)
    fake_image = torch.zeros(size=(3, *img_size), dtype=torch.uint8).numpy()
    fake_image_info = ImageInfo(img_idx=0, img_shape=img_size, ori_shape=img_size)
    fake_bboxes = tv_tensors.BoundingBoxes(data=torch.Tensor([0, 0, 5, 5]), format="xyxy", canvas_size=(10, 10))
    fake_labels = LongTensor([1])
    fake_masks = Mask(torch.randint(low=0, high=255, size=(1, *img_size), dtype=torch.uint8))
    fake_polygons = [Polygon(points=[1, 1, 2, 2, 3, 3, 4, 4])]
    # define data entity
    single_data_entity = InstanceSegDataEntity(
        image=fake_image,
        img_info=fake_image_info,
        bboxes=fake_bboxes,
        masks=fake_masks,
        labels=fake_labels,
        polygons=fake_polygons,
    )
    batch_data_entity = InstanceSegBatchDataEntity(
        batch_size=1,
        images=[Image(data=torch.from_numpy(fake_image))],
        imgs_info=[fake_image_info],
        bboxes=[fake_bboxes],
        labels=[fake_labels],
        masks=[fake_masks],
        polygons=[fake_polygons],
    )
    batch_pred_data_entity = InstanceSegBatchPredEntity(
        batch_size=1,
        images=[Image(data=torch.from_numpy(fake_image))],
        imgs_info=[fake_image_info],
        bboxes=[fake_bboxes],
        labels=[fake_labels],
        masks=[fake_masks],
        scores=[],
        polygons=[fake_polygons],
    )

    return single_data_entity, batch_pred_data_entity, batch_data_entity


@pytest.fixture(scope="session")
def fxt_seg_data_entity() -> tuple[tuple, SegDataEntity, SegBatchDataEntity]:
    img_size = (32, 32)
    fake_image = torch.zeros(size=(3, *img_size), dtype=torch.uint8).numpy()
    fake_image_info = ImageInfo(img_idx=0, img_shape=img_size, ori_shape=img_size)
    fake_masks = Mask(torch.randint(low=0, high=2, size=img_size, dtype=torch.uint8))
    # define data entity
    single_data_entity = SegDataEntity(
        image=fake_image,
        img_info=fake_image_info,
        masks=fake_masks,
    )
    batch_data_entity = SegBatchDataEntity(
        batch_size=1,
        images=[Image(data=torch.from_numpy(fake_image))],
        imgs_info=[fake_image_info],
        masks=[fake_masks],
    )
    batch_pred_data_entity = SegBatchPredEntity(
        batch_size=1,
        images=[Image(data=torch.from_numpy(fake_image))],
        imgs_info=[fake_image_info],
        masks=[fake_masks],
        scores=[],
    )

    return single_data_entity, batch_pred_data_entity, batch_data_entity


@pytest.fixture(autouse=True)
def fxt_clean_up_mem_cache():
    """Clean up the mem-cache instance at the end of the test.

    It is required for everyone who tests model training pipeline.
    See https://github.com/openvinotoolkit/training_extensions/actions/runs/7326689283/job/19952721142?pr=2749#step:5:3098
    """
    yield
    MemCacheHandlerSingleton.delete()


@pytest.fixture(scope="session")
def fxt_accelerator(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--device", "gpu")


@pytest.fixture(params=set(OTXTaskType) - {OTXTaskType.DETECTION_SEMI_SL})
def fxt_task(request: pytest.FixtureRequest) -> OTXTaskType:
    return request.param


@pytest.fixture(scope="session", autouse=True)
def fxt_null_label_info() -> LabelInfo:
    return NullLabelInfo()


@pytest.fixture(scope="session", autouse=True)
def fxt_seg_label_info() -> SegLabelInfo:
    label_names = ["class1", "class2", "class3"]
    return SegLabelInfo(
        label_names=label_names,
        label_groups=[
            label_names,
            ["class2", "class3"],
        ],
    )


@pytest.fixture(scope="session", autouse=True)
def fxt_multiclass_labelinfo() -> LabelInfo:
    label_names = ["class1", "class2", "class3"]
    return LabelInfo(
        label_names=label_names,
        label_groups=[
            label_names,
            ["class2", "class3"],
        ],
    )


@pytest.fixture(scope="session", autouse=True)
def fxt_multilabel_labelinfo() -> LabelInfo:
    label_names = ["class1", "class2", "class3"]
    return LabelInfo(
        label_names=label_names,
        label_groups=[
            [label_names[0]],
            [label_names[1]],
            [label_names[2]],
        ],
    )


@pytest.fixture()
def fxt_hlabel_multilabel_info() -> HLabelInfo:
    return HLabelInfo(
        label_names=[
            "Heart",
            "Spade",
            "Heart_Queen",
            "Heart_King",
            "Spade_A",
            "Spade_King",
            "Black_Joker",
            "Red_Joker",
            "Extra_Joker",
        ],
        label_groups=[
            ["Heart", "Spade"],
            ["Heart_Queen", "Heart_King"],
            ["Spade_A", "Spade_King"],
            ["Black_Joker"],
            ["Red_Joker"],
            ["Extra_Joker"],
        ],
        num_multiclass_heads=3,
        num_multilabel_classes=3,
        head_idx_to_logits_range={"0": (0, 2), "1": (2, 4), "2": (4, 6)},
        num_single_label_classes=3,
        empty_multiclass_head_indices=[],
        class_to_group_idx={
            "Heart": (0, 0),
            "Spade": (0, 1),
            "Heart_Queen": (1, 0),
            "Heart_King": (1, 1),
            "Spade_A": (2, 0),
            "Spade_King": (2, 1),
            "Black_Joker": (3, 0),
            "Red_Joker": (3, 1),
            "Extra_Joker": (3, 2),
        },
        all_groups=[
            ["Heart", "Spade"],
            ["Heart_Queen", "Heart_King"],
            ["Spade_A", "Spade_King"],
            ["Black_Joker"],
            ["Red_Joker"],
            ["Extra_Joker"],
        ],
        label_to_idx={
            "Heart": 0,
            "Spade": 1,
            "Heart_Queen": 2,
            "Heart_King": 3,
            "Spade_A": 4,
            "Spade_King": 5,
            "Black_Joker": 6,
            "Red_Joker": 7,
            "Extra_Joker": 8,
        },
        label_tree_edges=[
            ["Heart_Queen", "Heart"],
            ["Heart_King", "Heart"],
            ["Spade_A", "Spade"],
            ["Spade_King", "Spade"],
        ],
    )


@pytest.fixture()
def fxt_xpu_support_task() -> list[OTXTaskType]:
    return [
        OTXTaskType.ANOMALY_CLASSIFICATION,
        OTXTaskType.ANOMALY_DETECTION,
        OTXTaskType.ANOMALY_SEGMENTATION,
        OTXTaskType.MULTI_CLASS_CLS,
        OTXTaskType.MULTI_LABEL_CLS,
        OTXTaskType.H_LABEL_CLS,
        OTXTaskType.DETECTION,
        OTXTaskType.ROTATED_DETECTION,
        OTXTaskType.DETECTION_SEMI_SL,
        OTXTaskType.SEMANTIC_SEGMENTATION,
    ]


@pytest.fixture()
def fxt_export_list() -> list[ExportCase2Test]:
    return [
        ExportCase2Test("ONNX", False, "exported_model.onnx"),
        ExportCase2Test("OPENVINO", False, "exported_model.xml"),
        ExportCase2Test("OPENVINO", True, "exportable_code.zip"),
    ]
