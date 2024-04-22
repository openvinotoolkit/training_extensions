# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from otx.algo.classification.classifier import ImageClassifier
from otx.algo.classification.mobilenet_v3 import (
    MobileNetV3ForHLabelCls,
    MobileNetV3ForMulticlassCls,
    MobileNetV3ForMultilabelCls,
)
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.classification import (
    HlabelClsBatchPredEntity,
    MulticlassClsBatchPredEntity,
    MultilabelClsBatchPredEntity,
)


@pytest.fixture()
def fxt_multi_class_cls_model():
    return MobileNetV3ForMulticlassCls(
        mode="large",
        label_info=10,
    )


class TestMobileNetV3ForMulticlassCls:
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
        assert isinstance(preds, OTXBatchLossEntity)

        fxt_multi_class_cls_model.training = False
        preds = fxt_multi_class_cls_model._customize_outputs(outputs, fxt_multiclass_cls_batch_data_entity)
        assert isinstance(preds, MulticlassClsBatchPredEntity)

    @pytest.mark.parametrize("explain_mode", [True, False])
    def test_predict_step(self, fxt_multi_class_cls_model, fxt_multiclass_cls_batch_data_entity, explain_mode):
        fxt_multi_class_cls_model.eval()
        fxt_multi_class_cls_model.explain_mode = explain_mode
        outputs = fxt_multi_class_cls_model.predict_step(batch=fxt_multiclass_cls_batch_data_entity, batch_idx=0)

        assert isinstance(outputs, MulticlassClsBatchPredEntity)
        assert outputs.has_xai_outputs == explain_mode


@pytest.fixture()
def fxt_multi_label_cls_model():
    return MobileNetV3ForMultilabelCls(
        mode="large",
        label_info=10,
    )


class TestMobileNetV3ForMultilabelCls:
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
        assert isinstance(preds, OTXBatchLossEntity)

        fxt_multi_label_cls_model.training = False
        preds = fxt_multi_label_cls_model._customize_outputs(outputs, fxt_multilabel_cls_batch_data_entity)
        assert isinstance(preds, MultilabelClsBatchPredEntity)

    @pytest.mark.parametrize("explain_mode", [True, False])
    def test_predict_step(self, fxt_multi_label_cls_model, fxt_multilabel_cls_batch_data_entity, explain_mode):
        fxt_multi_label_cls_model.eval()
        fxt_multi_label_cls_model.explain_mode = explain_mode
        outputs = fxt_multi_label_cls_model.predict_step(batch=fxt_multilabel_cls_batch_data_entity, batch_idx=0)

        assert isinstance(outputs, MultilabelClsBatchPredEntity)
        assert outputs.has_xai_outputs == explain_mode


@pytest.fixture()
def fxt_h_label_cls_model(fxt_hlabel_data):
    return MobileNetV3ForHLabelCls(
        mode="large",
        label_info=fxt_hlabel_data,
    )


class TestMobileNetV3ForHLabelCls:
    def test_create_model(self, fxt_h_label_cls_model):
        assert isinstance(fxt_h_label_cls_model.model, ImageClassifier)

    def test_customize_inputs(self, fxt_h_label_cls_model, fxt_hlabel_cls_batch_data_entity):
        outputs = fxt_h_label_cls_model._customize_inputs(fxt_hlabel_cls_batch_data_entity)
        assert "images" in outputs
        assert "labels" in outputs
        assert "mode" in outputs

    def test_customize_outputs(self, fxt_h_label_cls_model, fxt_hlabel_cls_batch_data_entity):
        outputs = torch.randn(2, 10)
        fxt_h_label_cls_model.training = True
        preds = fxt_h_label_cls_model._customize_outputs(outputs, fxt_hlabel_cls_batch_data_entity)
        assert isinstance(preds, OTXBatchLossEntity)

        fxt_h_label_cls_model.training = False
        preds = fxt_h_label_cls_model._customize_outputs(outputs, fxt_hlabel_cls_batch_data_entity)
        assert isinstance(preds, HlabelClsBatchPredEntity)

    @pytest.mark.parametrize("explain_mode", [True, False])
    def test_predict_step(self, fxt_h_label_cls_model, fxt_hlabel_cls_batch_data_entity, explain_mode):
        fxt_h_label_cls_model.eval()
        fxt_h_label_cls_model.explain_mode = explain_mode
        outputs = fxt_h_label_cls_model.predict_step(batch=fxt_hlabel_cls_batch_data_entity, batch_idx=0)

        assert isinstance(outputs, HlabelClsBatchPredEntity)
        assert outputs.has_xai_outputs == explain_mode
