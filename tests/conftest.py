# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.segmentation import SegBatchDataEntity, SegBatchPredEntity, SegDataEntity
from otx.core.data.mem_cache import MemCacheHandlerSingleton
from otx.core.types.task import OTXTaskType
from torchvision.tv_tensors import Image, Mask


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
        "--summary-csv",
        action="store",
        help="Path to output summary cvs file. Defaults to {output-root}/benchmark-summary.csv",
    )
    parser.addoption(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print OTX commands without execution.",
    )
    parser.addoption(
        "--deterministic",
        action="store_true",
        default=False,
        help="Turn on deterministic training.",
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
        gt_seg_map=fake_masks,
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
def fxt_clean_up_mem_cache() -> None:
    """Clean up the mem-cache instance at the end of the test.

    It is required for everyone who tests model training pipeline.
    See https://github.com/openvinotoolkit/training_extensions/actions/runs/7326689283/job/19952721142?pr=2749#step:5:3098
    """
    yield
    MemCacheHandlerSingleton.delete()


# TODO(Jaeguk): Add cpu param when OTX can run integration test parallelly for each task.
@pytest.fixture(params=[pytest.param("gpu", marks=pytest.mark.gpu)])
def fxt_accelerator(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(params=set(OTXTaskType) - {OTXTaskType.DETECTION_SEMI_SL})
def fxt_task(request: pytest.FixtureRequest) -> OTXTaskType:
    return request.param
