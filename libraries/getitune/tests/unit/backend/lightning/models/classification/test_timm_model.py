# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.lightning.models.classification.classifier import ImageClassifier
from getitune.backend.lightning.models.classification.multiclass_models.timm_model import TimmModelMulticlassCls
from getitune.backend.lightning.models.classification.multilabel_models.timm_model import TimmModelMultilabelCls
from getitune.backend.lightning.models.classification.optimizers import TimmOptimizer
from getitune.data.entity.base import BatchLoss
from getitune.data.entity.sample import PredictionBatch


@pytest.fixture
def fxt_multi_class_cls_model():
    return TimmModelMulticlassCls(
        label_info=10,
        model_name="tf_efficientnetv2_s.in21k",
        data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        pretrained=False,
    )


class TestTimmModelMulticlassCls:
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

    @pytest.mark.parametrize(
        "explain_mode",
        [
            False,
            pytest.param(
                True,
                marks=pytest.mark.xfail(
                    reason="Explain mode expects spatial feature maps; timm backbones currently return pooled embeddings."
                ),
            ),
        ],
    )
    def test_predict_step(self, fxt_multi_class_cls_model, fxt_multiclass_cls_batch_data_entity, explain_mode):
        fxt_multi_class_cls_model.eval()
        fxt_multi_class_cls_model.explain_mode = explain_mode
        outputs = fxt_multi_class_cls_model.predict_step(batch=fxt_multiclass_cls_batch_data_entity, batch_idx=0)

        assert isinstance(outputs, PredictionBatch)
        assert (outputs.saliency_map is not None and len(outputs.saliency_map) > 0) == explain_mode

    def test_freeze_backbone(self):
        data_input_params = DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

        model = TimmModelMulticlassCls(
            label_info=10,
            model_name="tf_efficientnetv2_s.in21k",
            data_input_params=data_input_params,
            freeze_backbone=True,
            pretrained=False,
        )

        classification_layers = model._identify_classification_layers()
        assert all(param.requires_grad == (name in classification_layers) for name, param in model.named_parameters())

        model = TimmModelMulticlassCls(
            label_info=10,
            model_name="tf_efficientnetv2_s.in21k",
            data_input_params=data_input_params,
            freeze_backbone=False,
            pretrained=False,
        )
        assert all(param.requires_grad for param in model.parameters())

    def test_timm_optimizer_bound_to_model_name(self):
        optimizer = TimmOptimizer(lr=0.01, weight_decay=0.001)
        model = TimmModelMulticlassCls(
            label_info=10,
            model_name="vit_base_patch16_224",
            data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            optimizer=optimizer,
            pretrained=False,
        )
        assert model.optimizer_callable.model_name == "vit_base_patch16_224"  # pyrefly: ignore[missing-attribute]


@pytest.fixture
def fxt_multi_label_cls_model():
    return TimmModelMultilabelCls(
        label_info=10,
        model_name="tf_efficientnetv2_s.in21k",
        data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        pretrained=False,
    )


class TestTimmModelMultilabelCls:
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

    @pytest.mark.parametrize(
        "explain_mode",
        [
            False,
            pytest.param(
                True,
                marks=pytest.mark.xfail(
                    reason="Explain mode expects spatial feature maps; timm backbones currently return pooled embeddings."
                ),
            ),
        ],
    )
    def test_predict_step(self, fxt_multi_label_cls_model, fxt_multilabel_cls_batch_data_entity, explain_mode):
        fxt_multi_label_cls_model.eval()
        fxt_multi_label_cls_model.explain_mode = explain_mode
        outputs = fxt_multi_label_cls_model.predict_step(batch=fxt_multilabel_cls_batch_data_entity, batch_idx=0)

        assert isinstance(outputs, PredictionBatch)
        assert (outputs.saliency_map is not None and len(outputs.saliency_map) > 0) == explain_mode

    def test_freeze_backbone(self):
        data_input_params = DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

        model = TimmModelMultilabelCls(
            label_info=10,
            model_name="tf_efficientnetv2_s.in21k",
            data_input_params=data_input_params,
            freeze_backbone=True,
            pretrained=False,
        )

        classification_layers = model._identify_classification_layers()
        assert all(param.requires_grad == (name in classification_layers) for name, param in model.named_parameters())

        model = TimmModelMultilabelCls(
            label_info=10,
            model_name="tf_efficientnetv2_s.in21k",
            data_input_params=data_input_params,
            freeze_backbone=False,
            pretrained=False,
        )
        assert all(param.requires_grad for param in model.parameters())

    def test_timm_optimizer_bound_to_model_name(self):
        optimizer = TimmOptimizer(lr=0.01, weight_decay=0.001)
        model = TimmModelMultilabelCls(
            label_info=10,
            model_name="vit_base_patch16_224",
            data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            optimizer=optimizer,
            pretrained=False,
        )
        assert model.optimizer_callable.model_name == "vit_base_patch16_224"  # pyrefly: ignore[missing-attribute]
