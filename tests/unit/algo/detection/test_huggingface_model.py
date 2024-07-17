# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.detection import DetBatchDataEntity, DetBatchPredEntity
from torchvision import tv_tensors

SKIP_TRANSFORMERS_TEST = False
try:
    from otx.algo.detection.huggingface_model import HuggingFaceModelForDetection
    from transformers.models.detr.image_processing_detr import DetrImageProcessor
    from transformers.models.detr.modeling_detr import DetrForObjectDetection
    from transformers.utils.generic import ModelOutput
except ImportError:
    SKIP_TRANSFORMERS_TEST = True


@pytest.mark.skipif(SKIP_TRANSFORMERS_TEST, reason="'transformers' is not installed")
class TestHuggingFaceModelForDetection:
    @pytest.fixture()
    def fxt_detection_model(self):
        return HuggingFaceModelForDetection(
            model_name_or_path="facebook/detr-resnet-50",
            label_info=2,
        )

    @pytest.fixture()
    def fxt_det_batch_data_entity(self) -> DetBatchDataEntity:
        return DetBatchDataEntity(
            labels=[torch.tensor([0, 1]), torch.tensor([1, 1])],
            bboxes=[
                tv_tensors.BoundingBoxes(
                    torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]]),
                    format="XYXY",
                    canvas_size=(48, 48),
                ),
                tv_tensors.BoundingBoxes(
                    torch.tensor([[0, 0, 1, 1], [0.5, 0.5, 0.6, 0.6]]),
                    format="XYXY",
                    canvas_size=(48, 48),
                ),
            ],
            images=[torch.randn(3, 48, 48), torch.randn(3, 48, 48)],
            imgs_info=[
                ImageInfo(img_idx=1, img_shape=(48, 48), ori_shape=(48, 48)),
                ImageInfo(img_idx=2, img_shape=(48, 48), ori_shape=(48, 48)),
            ],
            batch_size=2,
        )

    def test_init(self, fxt_detection_model):
        assert isinstance(fxt_detection_model.model, DetrForObjectDetection)
        assert isinstance(fxt_detection_model.image_processor, DetrImageProcessor)

    def test_customize_inputs(self, fxt_detection_model, fxt_det_batch_data_entity):
        outputs = fxt_detection_model._customize_inputs(fxt_det_batch_data_entity)
        assert "pixel_values" in outputs
        assert "labels" in outputs
        assert len(outputs["labels"]) > 0
        assert "class_labels" in outputs["labels"][0]
        assert "boxes" in outputs["labels"][0]

    def test_customize_outputs(self, fxt_detection_model, fxt_det_batch_data_entity):
        outputs = ModelOutput(
            loss=torch.tensor(0.1),
            loss_dict={"ce_loss": torch.tensor(0.1), "bbox_loss": torch.tensor(0.1)},
            logits=torch.randn(2, 100, 2),
            pred_boxes=torch.randn(2, 100, 4),
        )
        fxt_detection_model.training = True
        result = fxt_detection_model._customize_outputs(outputs, fxt_det_batch_data_entity)
        assert isinstance(result, dict)
        assert "ce_loss" in result
        assert "bbox_loss" in result

        fxt_detection_model.training = False
        result = fxt_detection_model._customize_outputs(outputs, fxt_det_batch_data_entity)
        assert isinstance(result, DetBatchPredEntity)
        assert len(result.bboxes) == 2
        assert len(result.labels) == 2
        assert len(result.scores) == 2
