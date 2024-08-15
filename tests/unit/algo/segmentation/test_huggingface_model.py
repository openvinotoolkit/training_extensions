# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
import torch
from otx.core.data.entity.base import ImageInfo, OTXBatchLossEntity
from otx.core.data.entity.segmentation import SegBatchDataEntity, SegBatchPredEntity

SKIP_TRANSFORMERS_TEST = False
try:
    from otx.algo.segmentation import huggingface_model as target_file
    from otx.algo.segmentation.huggingface_model import HuggingFaceModelForSegmentation
    from transformers.modeling_outputs import SemanticSegmenterOutput
    from transformers.models.segformer.image_processing_segformer import SegformerImageProcessor
    from transformers.models.segformer.modeling_segformer import SegformerForSemanticSegmentation
except ImportError:
    SKIP_TRANSFORMERS_TEST = True


@pytest.mark.skipif(SKIP_TRANSFORMERS_TEST, reason="'transformers' is not installed")
class TestHuggingFaceModelForSegmentation:
    @pytest.fixture()
    def fxt_seg_model(self):
        return HuggingFaceModelForSegmentation(
            model_name_or_path="nvidia/segformer-b0-finetuned-ade-512-512",
            label_info=2,
        )

    @pytest.fixture()
    def fxt_seg_batch_data_entity(self) -> SegBatchDataEntity:
        return SegBatchDataEntity(
            images=[torch.randn(3, 32, 32), torch.randn(3, 32, 32)],
            masks=[
                torch.randint(low=0, high=2, size=(32, 32), dtype=torch.uint8),
                torch.randint(low=0, high=2, size=(32, 32), dtype=torch.uint8),
            ],
            imgs_info=[
                ImageInfo(img_idx=1, img_shape=(32, 32), ori_shape=(32, 32)),
                ImageInfo(img_idx=2, img_shape=(32, 32), ori_shape=(32, 32)),
            ],
            batch_size=2,
        )

    def test_init(self, fxt_seg_model):
        assert isinstance(fxt_seg_model.model, SegformerForSemanticSegmentation)
        assert isinstance(fxt_seg_model.image_processor, SegformerImageProcessor)

    def test_customize_inputs(self, fxt_seg_model, fxt_seg_batch_data_entity):
        outputs = fxt_seg_model._customize_inputs(fxt_seg_batch_data_entity)
        assert "pixel_values" in outputs
        assert "labels" in outputs
        assert isinstance(outputs["labels"], torch.Tensor)

    def test_customize_outputs(self, fxt_seg_model, fxt_seg_batch_data_entity):
        outputs = SemanticSegmenterOutput(
            loss=torch.tensor(0.1),
            logits=torch.randn(2, 2, 32, 32),
        )
        fxt_seg_model.training = True
        result = fxt_seg_model._customize_outputs(outputs, fxt_seg_batch_data_entity)
        assert isinstance(result, OTXBatchLossEntity)
        assert "loss" in result

        fxt_seg_model.training = False
        result = fxt_seg_model._customize_outputs(outputs, fxt_seg_batch_data_entity)
        assert isinstance(result, SegBatchPredEntity)
        assert result.batch_size == 2
        assert len(result.masks) == 2

        fxt_seg_model.explain_mode = True
        with pytest.raises(NotImplementedError):
            fxt_seg_model._customize_outputs(outputs, fxt_seg_batch_data_entity)

    @pytest.fixture()
    def mock_pretrainedconfig(self, mocker) -> MagicMock:
        mock_obj = mocker.patch.object(target_file, "PretrainedConfig")
        mock_obj.get_config_dict.return_value = ({"image_size": 512}, None)
        return mock_obj

    @pytest.fixture()
    def mock_automodel(self, mocker) -> MagicMock:
        return mocker.patch.object(target_file, "AutoModelForSemanticSegmentation")

    def test_set_input_size(self, mock_pretrainedconfig, mock_automodel):
        input_size = (1, 3, 1024, 1024)
        HuggingFaceModelForSegmentation(
            model_name_or_path="facebook/deit-tiny-patch16-224",
            label_info=10,
            input_size=input_size,
        )

        assert mock_automodel.from_pretrained.call_args.kwargs["image_size"] == input_size[-1]
