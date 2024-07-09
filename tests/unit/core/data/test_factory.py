# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test Factory classes for dataset and transforms."""

import pytest
from otx.core.config.data import SubsetConfig, VisualPromptingConfig
from otx.core.data.dataset.action_classification import OTXActionClsDataset
from otx.core.data.dataset.anomaly import AnomalyDataset
from otx.core.data.dataset.classification import (
    HLabelInfo,
    OTXHlabelClsDataset,
    OTXMulticlassClsDataset,
    OTXMultilabelClsDataset,
)
from otx.core.data.dataset.detection import OTXDetectionDataset
from otx.core.data.dataset.instance_segmentation import OTXInstanceSegDataset
from otx.core.data.dataset.segmentation import OTXSegmentationDataset
from otx.core.data.dataset.visual_prompting import OTXVisualPromptingDataset, OTXZeroShotVisualPromptingDataset
from otx.core.data.factory import OTXDatasetFactory, TransformLibFactory
from otx.core.data.transform_libs.torchvision import TorchVisionTransformLib
from otx.core.types.image import ImageColorChannel
from otx.core.types.task import OTXTaskType
from otx.core.types.transformer_libs import TransformLibType

lib_type_parameters = [(TransformLibType.TORCHVISION, TorchVisionTransformLib)]
SKIP_MMLAB_TEST = False
try:
    from otx.core.data.transform_libs.mmaction import MMActionTransformLib
    from otx.core.data.transform_libs.mmcv import MMCVTransformLib
    from otx.core.data.transform_libs.mmdet import MMDetTransformLib
    from otx.core.data.transform_libs.mmpretrain import MMPretrainTransformLib
    from otx.core.data.transform_libs.mmseg import MMSegTransformLib

    lib_type_parameters.extend(
        [
            (TransformLibType.MMCV, MMCVTransformLib),
            (TransformLibType.MMPRETRAIN, MMPretrainTransformLib),
            (TransformLibType.MMDET, MMDetTransformLib),
            (TransformLibType.MMSEG, MMSegTransformLib),
            (TransformLibType.MMACTION, MMActionTransformLib),
        ],
    )
except ImportError:
    SKIP_MMLAB_TEST = True


class TestTransformLibFactory:
    @pytest.mark.parametrize(
        ("lib_type", "lib"),
        lib_type_parameters,
    )
    def test_generate(self, lib_type, lib, mocker) -> None:
        mock_generate = mocker.patch.object(lib, "generate")
        config = mocker.MagicMock(spec=SubsetConfig)
        config.transform_lib_type = lib_type
        _ = TransformLibFactory.generate(config)
        mock_generate.assert_called_once_with(config)


class TestOTXDatasetFactory:
    @pytest.mark.parametrize(
        ("task_type", "dataset_cls"),
        [
            (OTXTaskType.MULTI_CLASS_CLS, OTXMulticlassClsDataset),
            (OTXTaskType.MULTI_LABEL_CLS, OTXMultilabelClsDataset),
            (OTXTaskType.H_LABEL_CLS, OTXHlabelClsDataset),
            (OTXTaskType.DETECTION, OTXDetectionDataset),
            (OTXTaskType.ROTATED_DETECTION, OTXInstanceSegDataset),
            (OTXTaskType.INSTANCE_SEGMENTATION, OTXInstanceSegDataset),
            (OTXTaskType.SEMANTIC_SEGMENTATION, OTXSegmentationDataset),
            (OTXTaskType.VISUAL_PROMPTING, OTXVisualPromptingDataset),
            (OTXTaskType.ZERO_SHOT_VISUAL_PROMPTING, OTXZeroShotVisualPromptingDataset),
            (OTXTaskType.ACTION_CLASSIFICATION, OTXActionClsDataset),
            (OTXTaskType.ANOMALY_CLASSIFICATION, AnomalyDataset),
            (OTXTaskType.ANOMALY_DETECTION, AnomalyDataset),
            (OTXTaskType.ANOMALY_SEGMENTATION, AnomalyDataset),
        ],
    )
    def test_create(
        self,
        fxt_mock_hlabelinfo,
        fxt_mock_dm_subset,
        fxt_mem_cache_handler,
        task_type,
        dataset_cls,
        mocker,
    ) -> None:
        mocker.patch.object(TransformLibFactory, "generate", return_value=None)
        cfg_subset = mocker.MagicMock(spec=SubsetConfig)
        vpm_config = mocker.MagicMock(spec=VisualPromptingConfig)
        vpm_config.use_bbox = False
        vpm_config.use_point = False
        image_color_channel = ImageColorChannel.BGR
        mocker.patch.object(HLabelInfo, "from_dm_label_groups", return_value=fxt_mock_hlabelinfo)
        assert isinstance(
            OTXDatasetFactory.create(
                task=task_type,
                dm_subset=fxt_mock_dm_subset,
                mem_cache_handler=fxt_mem_cache_handler,
                cfg_subset=cfg_subset,
                vpm_config=vpm_config,
                image_color_channel=image_color_channel,
            ),
            dataset_cls,
        )
