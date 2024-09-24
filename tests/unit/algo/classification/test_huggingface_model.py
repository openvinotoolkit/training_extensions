# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
import torch
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.classification import MulticlassClsBatchPredEntity
from otx.core.exporter.native import OTXNativeModelExporter

SKIP_TRANSFORMERS_TEST = False
try:
    from otx.algo.classification import huggingface_model as target_file
    from otx.algo.classification.huggingface_model import HuggingFaceModelForMulticlassCls
    from transformers.modeling_outputs import ImageClassifierOutput
except ImportError:
    SKIP_TRANSFORMERS_TEST = True


@pytest.mark.skipif(SKIP_TRANSFORMERS_TEST, reason="'transformers' is not installed")
class TestHuggingFaceModelForMulticlassCls:
    @pytest.fixture()
    def fxt_multi_class_cls_model(self):
        return HuggingFaceModelForMulticlassCls(
            model_name_or_path="facebook/deit-tiny-patch16-224",
            label_info=10,
        )

    def test_customize_inputs(self, fxt_multi_class_cls_model, fxt_multiclass_cls_batch_data_entity):
        outputs = fxt_multi_class_cls_model._customize_inputs(fxt_multiclass_cls_batch_data_entity)
        assert "pixel_values" in outputs
        assert "labels" in outputs

    def test_customize_outputs(self, fxt_multi_class_cls_model, fxt_multiclass_cls_batch_data_entity):
        outputs = ImageClassifierOutput(
            loss=torch.tensor(0.1),
            logits=torch.randn(2, 10),
            hidden_states=None,
            attentions=None,
        )
        fxt_multi_class_cls_model.training = True
        preds = fxt_multi_class_cls_model._customize_outputs(outputs, fxt_multiclass_cls_batch_data_entity)
        assert isinstance(preds, OTXBatchLossEntity)

        fxt_multi_class_cls_model.training = False
        preds = fxt_multi_class_cls_model._customize_outputs(outputs, fxt_multiclass_cls_batch_data_entity)
        assert isinstance(preds, MulticlassClsBatchPredEntity)

    def test_exporter(self, fxt_multi_class_cls_model):
        exporter = fxt_multi_class_cls_model._exporter
        assert isinstance(exporter, OTXNativeModelExporter)
        assert exporter.resize_mode == "standard"

    def test_forward_for_tracing(self, fxt_multi_class_cls_model, tmp_path):
        fxt_multi_class_cls_model.export(
            output_dir=tmp_path,
            base_name="exported_model",
            export_format="OPENVINO",
        )
        assert (tmp_path / "exported_model.xml").exists()

        output = fxt_multi_class_cls_model.forward_for_tracing(
            image=torch.randn(1, 3, 224, 224),
        )
        assert isinstance(output, ImageClassifierOutput)

        fxt_multi_class_cls_model.explain_mode = True
        with pytest.raises(NotImplementedError):
            output = fxt_multi_class_cls_model.forward_for_tracing(
                image=torch.randn(1, 3, 224, 224),
            )

    @pytest.fixture()
    def mock_pretrainedconfig(self, mocker) -> MagicMock:
        mock_obj = mocker.patch.object(target_file, "PretrainedConfig")
        mock_obj.get_config_dict.return_value = ({"image_size": 224}, None)
        return mock_obj

    @pytest.fixture()
    def mock_automodel(self, mocker) -> MagicMock:
        return mocker.patch.object(target_file, "AutoModelForImageClassification")

    def test_set_input_size(self, mock_pretrainedconfig, mock_automodel):
        input_size = (1, 3, 300, 300)
        HuggingFaceModelForMulticlassCls(
            model_name_or_path="facebook/deit-tiny-patch16-224",
            label_info=10,
            input_size=input_size,
        )

        assert mock_automodel.from_pretrained.call_args.kwargs["image_size"] == input_size[-1]
