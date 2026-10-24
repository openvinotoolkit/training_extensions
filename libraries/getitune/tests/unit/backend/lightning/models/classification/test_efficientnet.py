# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.lightning.models.classification.classifier import ImageClassifier
from getitune.backend.lightning.models.classification.multiclass_models.efficientnet import EfficientNetMulticlassCls
from getitune.backend.lightning.models.classification.multilabel_models.efficientnet import EfficientNetMultilabelCls
from getitune.data.entity.base import BatchLoss
from getitune.data.entity.sample import PredictionBatch


@pytest.fixture
def fxt_multi_class_cls_model():
    return EfficientNetMulticlassCls(
        model_name="efficientnet_b0",
        label_info=10,
        data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    )


class TestEfficientNetForMulticlassCls:
    def test_create_model(self, fxt_multi_class_cls_model):
        assert isinstance(fxt_multi_class_cls_model.model, ImageClassifier)

    def test_customize_inputs(self, fxt_multi_class_cls_model, fxt_multiclass_cls_batch_data_entity):
        outputs = fxt_multi_class_cls_model._customize_inputs(fxt_multiclass_cls_batch_data_entity)
        assert "images" in outputs
        assert "labels" in outputs
        assert "mode" in outputs

    def test_customize_outputs(self, fxt_multi_class_cls_model, fxt_multiclass_cls_batch_data_entity):
        outputs = torch.randn(2, 10)
        fxt_multi_class_cls_model.training = True
        preds = fxt_multi_class_cls_model._customize_outputs(outputs, fxt_multiclass_cls_batch_data_entity)
        assert isinstance(preds, BatchLoss)

        fxt_multi_class_cls_model.training = False
        preds = fxt_multi_class_cls_model._customize_outputs(outputs, fxt_multiclass_cls_batch_data_entity)
        assert isinstance(preds, PredictionBatch)

    @pytest.mark.parametrize("explain_mode", [True, False])
    def test_predict_step(self, fxt_multi_class_cls_model, fxt_multiclass_cls_batch_data_entity, explain_mode):
        fxt_multi_class_cls_model.eval()
        fxt_multi_class_cls_model.explain_mode = explain_mode
        outputs = fxt_multi_class_cls_model.predict_step(batch=fxt_multiclass_cls_batch_data_entity, batch_idx=0)

        assert isinstance(outputs, PredictionBatch)
        assert (outputs.saliency_map is not None and len(outputs.saliency_map) > 0) == explain_mode

    def test_set_input_size(self):
        data_input_params = DataInputParams((300, 300), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        model = EfficientNetMulticlassCls(
            model_name="efficientnet_b0",
            label_info=10,
            data_input_params=data_input_params,
        )
        input_size = data_input_params.input_size
        assert input_size is not None
        assert model.model.backbone.in_size == input_size[-2:]

    def test_freeze_backbone(self):
        data_input_params = DataInputParams((300, 300), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

        model = EfficientNetMulticlassCls(
            model_name="efficientnet_b0",
            label_info=10,
            data_input_params=data_input_params,
            freeze_backbone=True,
        )

        classification_layers = model._identify_classification_layers()
        assert all(param.requires_grad == (name in classification_layers) for name, param in model.named_parameters())

        model = EfficientNetMulticlassCls(
            model_name="efficientnet_b0",
            label_info=10,
            data_input_params=data_input_params,
            freeze_backbone=False,
        )

        assert all(param.requires_grad for param in model.parameters())


@pytest.fixture
def fxt_multi_label_cls_model():
    return EfficientNetMultilabelCls(
        model_name="efficientnet_b0",
        label_info=10,
        data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    )


class TestEfficientNetForMultilabelCls:
    def test_create_model(self, fxt_multi_label_cls_model):
        assert isinstance(fxt_multi_label_cls_model.model, ImageClassifier)

    def test_customize_inputs(self, fxt_multi_label_cls_model, fxt_multilabel_cls_batch_data_entity):
        outputs = fxt_multi_label_cls_model._customize_inputs(fxt_multilabel_cls_batch_data_entity)
        assert "images" in outputs
        assert "labels" in outputs
        assert "mode" in outputs

    def test_customize_outputs(self, fxt_multi_label_cls_model, fxt_multilabel_cls_batch_data_entity):
        outputs = torch.randn(2, 10)
        fxt_multi_label_cls_model.training = True
        preds = fxt_multi_label_cls_model._customize_outputs(outputs, fxt_multilabel_cls_batch_data_entity)
        assert isinstance(preds, BatchLoss)

        fxt_multi_label_cls_model.training = False
        preds = fxt_multi_label_cls_model._customize_outputs(outputs, fxt_multilabel_cls_batch_data_entity)
        assert isinstance(preds, PredictionBatch)

    @pytest.mark.parametrize("explain_mode", [True, False])
    def test_predict_step(self, fxt_multi_label_cls_model, fxt_multilabel_cls_batch_data_entity, explain_mode):
        fxt_multi_label_cls_model.eval()
        fxt_multi_label_cls_model.explain_mode = explain_mode
        outputs = fxt_multi_label_cls_model.predict_step(batch=fxt_multilabel_cls_batch_data_entity, batch_idx=0)

        assert isinstance(outputs, PredictionBatch)
        assert (outputs.saliency_map is not None and len(outputs.saliency_map) > 0) == explain_mode

    def test_set_input_size(self):
        data_input_params = DataInputParams((300, 300), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        model = EfficientNetMultilabelCls(
            model_name="efficientnet_b0",
            label_info=10,
            data_input_params=data_input_params,
        )
        input_size = data_input_params.input_size
        assert input_size is not None
        assert model.model.backbone.in_size == input_size[-2:]

    def test_freeze_backbone(self):
        data_input_params = DataInputParams((300, 300), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

        model = EfficientNetMultilabelCls(
            model_name="efficientnet_b0",
            label_info=10,
            data_input_params=data_input_params,
            freeze_backbone=True,
        )

        classification_layers = model._identify_classification_layers()
        assert all(param.requires_grad == (name in classification_layers) for name, param in model.named_parameters())

        model = EfficientNetMultilabelCls(
            model_name="efficientnet_b0",
            label_info=10,
            data_input_params=data_input_params,
            freeze_backbone=False,
        )

        assert all(param.requires_grad for param in model.parameters())
