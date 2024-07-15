# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of custom algo modules of OTX Detection task."""
import pytest
from otx.core.config.data import SubsetConfig
from otx.core.data.module import OTXDataModule
from otx.core.types.task import OTXTaskType
from torchvision.transforms.v2 import Resize


@pytest.fixture()
def fxt_data_module():
    return OTXDataModule(
        task=OTXTaskType.DETECTION,
        data_format="coco_instances",
        data_root="tests/assets/car_tree_bug",
        train_subset=SubsetConfig(
            batch_size=2,
            subset_name="train",
            transforms=[Resize(320)],
        ),
        val_subset=SubsetConfig(
            batch_size=2,
            subset_name="val",
            transforms=[Resize(320)],
        ),
        test_subset=SubsetConfig(
            batch_size=2,
            subset_name="test",
            transforms=[Resize(320)],
        ),
    )
