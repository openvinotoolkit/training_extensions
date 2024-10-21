# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test of custom algo modules of OTX Object Detection 3D task."""

import pytest
import torch
from otx.core.config.data import SubsetConfig
from otx.core.data.module import OTXDataModule
from otx.core.data.transform_libs.torchvision import Decode3DInputsAffineTransforms
from otx.core.types.task import OTXTaskType
from torchvision.transforms.v2 import Normalize, ToDtype


@pytest.fixture()
def fxt_data_module_3d():
    return OTXDataModule(
        task=OTXTaskType.OBJECT_DETECTION_3D,
        data_format="kitti3d",
        data_root="tests/assets/kitti3d",
        train_subset=SubsetConfig(
            batch_size=2,
            subset_name="train",
            transforms=[
                Decode3DInputsAffineTransforms((380, 1280), True),
                ToDtype(torch.float),
                Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
            ],
            to_tv_image=False,
        ),
        val_subset=SubsetConfig(
            batch_size=2,
            subset_name="val",
            transforms=[
                Decode3DInputsAffineTransforms((380, 1280), decode_annotations=False),
                ToDtype(torch.float),
                Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
            ],
            to_tv_image=False,
        ),
        test_subset=SubsetConfig(
            batch_size=2,
            subset_name="test",
            transforms=[
                Decode3DInputsAffineTransforms((380, 1280), decode_annotations=False),
                ToDtype(torch.float),
                Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
            ],
            to_tv_image=False,
        ),
    )
