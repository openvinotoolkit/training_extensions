# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import multiprocessing
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import torch
from torch import LongTensor
from torch.utils._pytree import register_pytree_node
from torchvision import tv_tensors
from torchvision.tv_tensors import Mask

from getitune.data.entity.base import ImageInfo
from getitune.data.entity.sample import PredictionBatch, SampleBatch
from getitune.types.label import LabelInfo, NullLabelInfo, SegLabelInfo
from getitune.types.task import TaskType
from getitune.utils.device import is_xpu_available
from tests.utils import ExportCase2Test

if TYPE_CHECKING:
    import numpy as np


@dataclass
class MockSample:
    """Mock sample class for testing purposes.

    This is a simple dataclass that mimics the BaseSample interface for tests.
    """

    image: torch.Tensor | np.ndarray
    img_info: ImageInfo | None = None
    label: torch.Tensor | None = None
    masks: Any | None = None
    bboxes: tv_tensors.BoundingBoxes | None = None
    keypoints: torch.Tensor | None = None


def _mocksample_flatten(sample: MockSample) -> tuple[list[Any], dict[str, Any]]:
    """Flatten MockSample for pytree traversal."""
    values = [getattr(sample, f.name) for f in fields(sample)]
    context = {f.name: None for f in fields(sample)}
    return values, context


def _mocksample_unflatten(values: list[Any], context: dict[str, Any]) -> MockSample:
    """Unflatten values back into MockSample."""
    field_names = list(context.keys())
    return MockSample(**dict(zip(field_names, values)))


# Register MockSample with pytree so torchvision v2 transforms can traverse it
register_pytree_node(MockSample, _mocksample_flatten, _mocksample_unflatten)


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Initialize test environment safely to prevent segfaults.

    This fixture:
    - Sets multiprocessing method to 'spawn'
    - Disables CUDA for unit tests
    - Cleans up PyTorch internals on teardown
    - Ensures tests run from the correct directory (library root)
    """
    import os

    # Save the current working directory
    original_cwd = Path.cwd()

    # Find the library root directory (where pyproject.toml is located)
    tests_dir = Path(__file__).parent
    library_root = tests_dir.parent

    # Change to library root if not already there
    if Path.cwd() != library_root:
        os.chdir(library_root)

    # Set multiprocessing method if not already set
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method("spawn", force=True)

    yield

    # Restore original working directory
    os.chdir(original_cwd)

    # Force cleanup to prevent segfaults during pytest teardown
    try:
        torch._C._jit_clear_class_registry()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: S110
        pass  # Ignore cleanup errors


def pytest_addoption(parser: pytest):
    """Add custom options for perf tests."""
    parser.addoption(
        "--model-category",
        action="store",
        default="all",
        choices=("speed", "balance", "accuracy", "default", "other", "all"),
        help="Choose speed|balance|accuracy|default|other|all. Defaults to all.",
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
        help="Overrides default per-model number of epoch setting. Defaults to 0 (per-model epoch & early-stopping).",
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
        help="Print getitune commands without execution.",
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
        "--getitune-ref",
        type=str,
        default="__CURRENT_BRANCH_COMMIT__",
        help="Target getitune ref (tag / branch name / commit hash) on main repo to test. Defaults to the current branch. "
        "`pip install getitune[full]@https://github.com/open-edge-platform/training_extensions.git@{getitune_ref}` will be executed before run, "
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
        help="Task type of getitune to use test.",
    )
    parser.addoption(
        "--device",
        action="store",
        default="gpu",
        type=str,
        help="Which device to use.",
    )
    parser.addoption(
        "--run-category-only",
        action="store_true",
        help="Run only the model category tests that categorised as BALANCE, SPEED, ACCURACY.",
    )
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="Run tests marked with @pytest.mark.network that perform real network I/O. "
        "Skipped by default to keep CI fast and offline-friendly.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip ``@pytest.mark.network`` tests unless ``--run-network`` is given."""
    if config.getoption("--run-network"):
        return
    skip_network = pytest.mark.skip(reason="needs --run-network to run (network I/O)")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)


@pytest.fixture(scope="session")
def fxt_multi_class_cls_data_entity() -> tuple[MockSample, SampleBatch, SampleBatch]:
    img_size = (64, 64)
    fake_images = torch.zeros(size=(1, 3, *img_size), dtype=torch.float32)
    fake_image_info = ImageInfo(img_idx=0, img_shape=img_size, ori_shape=img_size)
    fake_labels = LongTensor([0])
    fake_score = torch.Tensor([0.6])
    # define data entity
    single_data_entity = MockSample(image=fake_images[0], img_info=fake_image_info, label=fake_labels)
    batch_data_entity = SampleBatch(
        images=fake_images,
        imgs_info=[fake_image_info],
        labels=[fake_labels],
    )
    batch_pred_data_entity = PredictionBatch(
        images=fake_images,
        imgs_info=[fake_image_info],
        labels=[fake_labels],
        scores=[fake_score],
    )

    return single_data_entity, batch_pred_data_entity, batch_data_entity


@pytest.fixture(scope="session")
def fxt_multi_label_cls_data_entity() -> tuple[MockSample, SampleBatch, SampleBatch]:
    img_size = (64, 64)
    fake_images = torch.zeros(size=(1, 3, *img_size), dtype=torch.float32)
    fake_image_info = ImageInfo(img_idx=0, img_shape=img_size, ori_shape=img_size)
    fake_labels = LongTensor([0])
    fake_score = torch.Tensor([0.6])
    # define data entity
    single_data_entity = MockSample(image=fake_images[0], img_info=fake_image_info, label=fake_labels)
    batch_data_entity = SampleBatch(
        images=fake_images,
        imgs_info=[fake_image_info],
        labels=[fake_labels],
    )
    batch_pred_data_entity = PredictionBatch(
        images=fake_images,
        imgs_info=[fake_image_info],
        labels=[fake_labels],
        scores=[fake_score],
    )

    return single_data_entity, batch_pred_data_entity, batch_data_entity


@pytest.fixture(scope="session")
def fxt_det_data_entity() -> tuple[tuple, MockSample, SampleBatch]:
    img_size = (64, 64)
    fake_image = torch.zeros(size=(3, *img_size), dtype=torch.float32)
    fake_images = fake_image.unsqueeze(0)  # (1, 3, H, W)
    fake_image_info = ImageInfo(img_idx=0, img_shape=img_size, ori_shape=img_size)
    fake_bboxes = tv_tensors.BoundingBoxes(data=torch.Tensor([0, 0, 5, 5]), format="xyxy", canvas_size=(10, 10))
    fake_labels = LongTensor([1])
    # define data entity
    single_data_entity = MockSample(
        image=fake_image,
        img_info=fake_image_info,
        bboxes=fake_bboxes,
        label=fake_labels,
    )
    batch_data_entity = SampleBatch(
        images=fake_images,
        imgs_info=[fake_image_info],
        bboxes=[fake_bboxes],
        labels=[fake_labels],
    )
    batch_pred_data_entity = PredictionBatch(
        images=fake_images,
        imgs_info=[fake_image_info],
        bboxes=[fake_bboxes],
        labels=[fake_labels],
        scores=[],
    )

    return single_data_entity, batch_pred_data_entity, batch_data_entity


@pytest.fixture(scope="session")
def fxt_inst_seg_data_entity() -> tuple[tuple, MockSample, SampleBatch]:
    img_size = (64, 64)
    fake_image = torch.zeros(size=(3, *img_size), dtype=torch.float32)
    fake_images = fake_image.unsqueeze(0)  # (1, 3, H, W)
    fake_image_info = ImageInfo(img_idx=0, img_shape=img_size, ori_shape=img_size)
    fake_bboxes = tv_tensors.BoundingBoxes(data=torch.Tensor([0, 0, 5, 5]), format="xyxy", canvas_size=(10, 10))
    fake_labels = LongTensor([1])
    fake_masks = Mask(torch.randint(low=0, high=255, size=(1, *img_size), dtype=torch.uint8))

    # define data entity
    single_data_entity = MockSample(
        image=fake_image,
        img_info=fake_image_info,
        bboxes=fake_bboxes,
        masks=fake_masks,
        label=fake_labels,
    )
    batch_data_entity = SampleBatch(
        images=fake_images,
        imgs_info=[fake_image_info],
        bboxes=[fake_bboxes],
        labels=[fake_labels],
        masks=[fake_masks],
    )
    batch_pred_data_entity = PredictionBatch(
        images=fake_images,
        imgs_info=[fake_image_info],
        bboxes=[fake_bboxes],
        labels=[fake_labels],
        masks=[fake_masks],
    )

    return single_data_entity, batch_pred_data_entity, batch_data_entity


@pytest.fixture(scope="session")
def fxt_seg_data_entity() -> tuple[tuple, MockSample, SampleBatch]:
    img_size = (32, 32)
    fake_image = torch.zeros(size=(3, *img_size), dtype=torch.float32)
    fake_images = fake_image.unsqueeze(0)  # (1, 3, H, W)
    fake_image_info = ImageInfo(img_idx=0, img_shape=img_size, ori_shape=img_size)
    fake_masks = Mask(torch.randint(low=0, high=2, size=img_size, dtype=torch.uint8))
    # define data entity
    single_data_entity = MockSample(
        image=fake_image,
        img_info=fake_image_info,
        masks=fake_masks,
    )
    batch_data_entity = SampleBatch(
        images=fake_images,
        imgs_info=[fake_image_info],
        masks=[fake_masks],
    )
    batch_pred_data_entity = PredictionBatch(
        images=fake_images,
        imgs_info=[fake_image_info],
        masks=[fake_masks],
        scores=[],
    )

    return single_data_entity, batch_pred_data_entity, batch_data_entity


@pytest.fixture(scope="session")
def fxt_accelerator(request: pytest.FixtureRequest) -> str:
    if is_xpu_available():
        return "xpu"
    return request.config.getoption("--device", "gpu")


@pytest.fixture(params=set(TaskType))
def fxt_task(request: pytest.FixtureRequest) -> TaskType:
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
        label_ids=["0", "1", "2"],
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
        label_ids=["0", "1", "2"],
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
        label_ids=["0", "1", "2"],
    )


@pytest.fixture
def fxt_export_list() -> list[ExportCase2Test]:
    return [
        ExportCase2Test("ONNX", False, "exported_model.onnx"),
        ExportCase2Test("OPENVINO", False, "exported_model.xml"),
        ExportCase2Test("OPENVINO", True, "exportable_code.zip"),
    ]
