# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test of custom algo modules of OTX Object Detection 3D task."""

import pytest
from otx.core.config.data import SubsetConfig
from otx.core.data.module import OTXDataModule
from otx.core.types.task import OTXTaskType
from torchvision.transforms.v2 import Normalize


@pytest.fixture()
def fxt_data_module_3d():
    return OTXDataModule(
        task=OTXTaskType.OBJECT_DETECTION_3D,
        data_format="kitti3d",
        data_root="tests/assets/kitti3d",
        train_subset=SubsetConfig(
            batch_size=2,
            subset_name="train",
            transforms=[Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],
            to_tv_image=False,
        ),
        val_subset=SubsetConfig(
            batch_size=2,
            subset_name="val",
            transforms=[Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],
            to_tv_image=False,
        ),
        test_subset=SubsetConfig(
            batch_size=2,
            subset_name="test",
            transforms=[Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],
            to_tv_image=False,
        ),
    )
