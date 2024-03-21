"""Test OTX models."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from otx.algo.anomaly.draem import Draem
from otx.algo.anomaly.padim import Padim
from otx.algo.anomaly.stfpm import Stfpm
from otx.core.data.entity.anomaly import (
    AnomalyClassificationBatchPrediction,
    AnomalyClassificationDataBatch,
    AnomalyDetectionBatchPrediction,
    AnomalyDetectionDataBatch,
    AnomalySegmentationBatchPrediction,
    AnomalySegmentationDataBatch,
)
from otx.core.data.entity.base import ImageInfo
from otx.core.model.entity.base import OTXModel
from otx.core.model.module.anomaly import AnomalyModelInputs, OTXAnomaly
from otx.core.types.task import OTXTaskType
from torchvision import tv_tensors


class _DummyModel(OTXAnomaly, OTXModel):
    """Dummy model for testing."""

    def __init__(self):
        OTXAnomaly.__init__(self)
        OTXModel.__init__(self, num_classes=2)


class TestAnomalyModel:
    def _get_batch(self, task: OTXTaskType) -> AnomalyModelInputs:
        """Get batch for testing based on the task type."""
        image = tv_tensors.Image(data=torch.rand(3, 8, 8))
        img_info = ImageInfo(img_idx=0, img_shape=(8, 8), ori_shape=(8, 8))
        if task == OTXTaskType.ANOMALY_CLASSIFICATION:
            batch = AnomalyClassificationDataBatch(
                batch_size=1,
                images=[image],
                imgs_info=img_info,
                labels=[torch.tensor(0)],
            )
        elif task == OTXTaskType.ANOMALY_DETECTION:
            batch = AnomalyDetectionDataBatch(
                batch_size=1,
                images=[image],
                imgs_info=img_info,
                labels=[torch.tensor(0)],
                boxes=[torch.tensor([0, 0, 8, 8])],
                masks=torch.rand(1, 8, 8),
            )
        elif task == OTXTaskType.ANOMALY_SEGMENTATION:
            batch = AnomalySegmentationDataBatch(
                batch_size=1,
                images=[image],
                imgs_info=img_info,
                labels=[torch.tensor(0)],
                masks=torch.rand(1, 8, 8),
            )
        return batch

    @pytest.mark.parametrize("model", [Draem, Padim, Stfpm])
    def test_model_instantiation(self, model):
        """Test if the model can be instantiated."""
        anomaly_model = model()
        assert isinstance(anomaly_model, OTXModel)

    @pytest.mark.parametrize(
        "task",
        [OTXTaskType.ANOMALY_CLASSIFICATION, OTXTaskType.ANOMALY_DETECTION, OTXTaskType.ANOMALY_SEGMENTATION],
    )
    def test_customize_inputs_outputs(self, task):
        """Test if the inputs are converted correctly."""
        batch = self._get_batch(task)
        anomaly_model = _DummyModel()
        anomaly_model.task = task
        output = anomaly_model._customize_inputs(batch)

        if task == OTXTaskType.ANOMALY_CLASSIFICATION:
            assert set(output.keys()) == {"image", "label"}
        elif task == OTXTaskType.ANOMALY_DETECTION:
            assert set(output.keys()) == {"image", "label", "mask", "boxes"}
        else:
            assert set(output.keys()) == {"image", "label", "mask"}

    @pytest.mark.parametrize(
        "task",
        [OTXTaskType.ANOMALY_CLASSIFICATION, OTXTaskType.ANOMALY_DETECTION, OTXTaskType.ANOMALY_SEGMENTATION],
    )
    def test_customize_outputs(self, task):
        """Test if the outputs are converted correctly."""
        batch = self._get_batch(task)
        anomaly_model = _DummyModel()
        anomaly_model.task = task
        output_dict = {
            "label": torch.tensor([0]),
            "pred_scores": torch.tensor([0.5]),
            "anomaly_maps": torch.rand(1, 8, 8),
            "mask": torch.rand(1, 8, 8),
            "pred_boxes": torch.tensor([[0, 0, 8, 8]]),
            "box_scores": torch.tensor([0.5]),
            "box_labels": torch.tensor([0]),
        }
        output = anomaly_model._customize_outputs(output_dict, inputs=batch)

        if task == OTXTaskType.ANOMALY_CLASSIFICATION:
            assert isinstance(output, AnomalyClassificationBatchPrediction)
        elif task == OTXTaskType.ANOMALY_DETECTION:
            assert isinstance(output, AnomalyDetectionBatchPrediction)
        else:
            assert isinstance(output, AnomalySegmentationBatchPrediction)
