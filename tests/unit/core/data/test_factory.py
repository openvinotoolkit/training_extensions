# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test Factory classes for dataset and transforms."""

from otx.core.config.data import DataModuleConfig, SubsetConfig
from otx.core.data.dataset.classification import OTXMulticlassClsDataset
from otx.core.data.dataset.detection import OTXDetectionDataset
from otx.core.data.dataset.segmentation import OTXSegmentationDataset
from otx.core.data.factory import OTXDatasetFactory, TransformLibFactory
from otx.core.data.transform_libs.mmcv import MMCVTransformLib
from otx.core.data.transform_libs.mmdet import MMDetTransformLib
from otx.core.data.transform_libs.mmpretrain import MMPretrainTransformLib
from otx.core.data.transform_libs.mmseg import MMSegTransformLib
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

        mocker.patch.object(MMSegTransformLib, "generate", return_value="MMSegTransforms")
        config.transform_lib_type=TransformLibType.MMSEG
        assert TransformLibFactory.generate(config) == "MMSegTransforms"


class TestOTXDatasetFactory:
    def test_create(self, mocker) -> None:
        def mock_convert_func(cfg: dict) -> dict:
            return cfg
        cfg_subset = SubsetConfig(
            batch_size=64,
            subset_name="train",
            transform_lib_type=TransformLibType.MMPRETRAIN,
            transforms=[{"type": "LoadImageFromFile"}, {"type": "PackInputs"}],
            num_workers=2,
        )
        cfg_data_module = DataModuleConfig(
            data_format="dummy",
            data_root="dummy",
            train_subset=cfg_subset,
            val_subset=cfg_subset,
            test_subset=cfg_subset,
            mem_cache_img_max_size=(512, 512),
        )
        mocker.patch.object(OTXMulticlassClsDataset, "__init__", return_value=None)
        mocker.patch("otx.core.data.transform_libs.mmcv.convert_conf_to_mmconfig_dict", side_effect=mock_convert_func)
        assert isinstance(
            OTXDatasetFactory.create(
                task=OTXTaskType.MULTI_CLASS_CLS, dm_subset=None, cfg_subset=cfg_subset, cfg_data_module=cfg_data_module,
            ),
            OTXMulticlassClsDataset,
        )

        cfg_subset = SubsetConfig(
            batch_size=64,
            subset_name="train",
            transform_lib_type=TransformLibType.MMDET,
            transforms=[{"type": "LoadImageFromFile"}, {"type": "LoadAnnotations"}, {"type": "PackDetInputs"}],
            num_workers=2,
        )
        mocker.patch.object(OTXDetectionDataset, "__init__", return_value=None)
        mocker.patch("otx.core.data.transform_libs.mmcv.convert_conf_to_mmconfig_dict", side_effect=mock_convert_func)
        assert isinstance(
            OTXDatasetFactory.create(
                task=OTXTaskType.DETECTION, dm_subset=None, cfg_subset=cfg_subset, cfg_data_module=cfg_data_module,
            ),
            OTXDetectionDataset,
        )

        cfg_subset = SubsetConfig(
            batch_size=64,
            subset_name="train",
            transform_lib_type=TransformLibType.MMSEG,
            transforms=[{"type": "LoadImageFromFile"}, {"type": "LoadAnnotations"}, {"type": "PackSegInputs"}],
            num_workers=2,
        )
        mocker.patch.object(OTXSegmentationDataset, "__init__", return_value=None)
        mocker.patch("otx.core.data.transform_libs.mmcv.convert_conf_to_mmconfig_dict", side_effect=mock_convert_func)
        assert isinstance(
            OTXDatasetFactory.create(
                task=OTXTaskType.SEMANTIC_SEGMENTATION, dm_subset=None, cfg_subset=cfg_subset, cfg_data_module=cfg_data_module,
            ),
            OTXSegmentationDataset,
        )
