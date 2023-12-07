# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test Factory classes for dataset and transforms."""

from otx.core.config.data import SubsetConfig
from otx.core.data.dataset.classification import OTXMulticlassClsDataset
from otx.core.data.dataset.detection import OTXDetectionDataset
from otx.core.data.factory import OTXDatasetFactory, TransformLibFactory
from otx.core.data.transform_libs.mmcv import MMCVTransformLib
from otx.core.data.transform_libs.mmdet import MMDetTransformLib
from otx.core.data.transform_libs.mmpretrain import MMPretrainTransformLib
from otx.core.types.task import OTXTaskType
from otx.core.types.transformer_libs import TransformLibType


class TestTransformLibFactory:
    def test_generate(self, mocker) -> None:
        mocker.patch.object(MMCVTransformLib, "generate", return_value="MMCVTransforms")
        config = SubsetConfig(
            batch_size=64,
            subset_name="train",
            transform_lib_type=TransformLibType.MMCV,
            transforms=[{"type": "LoadImageFromFile"}, {"type": "Normalize", "mean": [0, 0, 0], "std": [1, 1, 1]}],
            num_workers=2,
        )
        assert TransformLibFactory.generate(config) == "MMCVTransforms"

        mocker.patch.object(MMPretrainTransformLib, "generate", return_value="MMPretrainTransforms")
        config.transform_lib_type=TransformLibType.MMPRETRAIN
        assert TransformLibFactory.generate(config) == "MMPretrainTransforms"

        mocker.patch.object(MMDetTransformLib, "generate", return_value="MMDetTransforms")
        config.transform_lib_type=TransformLibType.MMDET
        assert TransformLibFactory.generate(config) == "MMDetTransforms"


class TestOTXDatasetFactory:
    def test_create(self, mocker) -> None:
        def mock_convert_func(cfg: dict) -> dict:
            return cfg
        config = SubsetConfig(
            batch_size=64,
            subset_name="train",
            transform_lib_type=TransformLibType.MMPRETRAIN,
            transforms=[{"type": "LoadImageFromFile"}, {"type": "PackInputs"}],
            num_workers=2,
        )
        mocker.patch.object(OTXMulticlassClsDataset, "__init__", return_value=None)
        mocker.patch("otx.core.data.transform_libs.mmcv.convert_conf_to_mmconfig_dict", side_effect=mock_convert_func)
        assert isinstance(
            OTXDatasetFactory.create(task=OTXTaskType.MULTI_CLASS_CLS, dm_subset=None, config=config), OTXMulticlassClsDataset,
        )

        config = SubsetConfig(
            batch_size=64,
            subset_name="train",
            transform_lib_type=TransformLibType.MMDET,
            transforms=[{"type": "LoadImageFromFile"}, {"type": "LoadAnnotations"}, {"type": "PackDetInputs"}],
            num_workers=2,
        )
        mocker.patch.object(OTXDetectionDataset, "__init__", return_value=None)
        mocker.patch("otx.core.data.transform_libs.mmcv.convert_conf_to_mmconfig_dict", side_effect=mock_convert_func)
        assert isinstance(
            OTXDatasetFactory.create(task=OTXTaskType.DETECTION, dm_subset=None, config=config), OTXDetectionDataset,
        )
