# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from otx.algo.classification.classifier import ImageClassifier
from otx.algo.classification.heads import SemiSLLinearClsHead
from otx.algo.classification.torchvision_model import (
    TVModelForHLabelCls,
    TVModelForMulticlassCls,
    TVModelForMultilabelCls,
)
from otx.core.data.entity.base import OTXBatchLossEntity, OTXBatchPredEntity
from otx.core.data.entity.classification import MulticlassClsBatchPredEntity
from otx.core.types.export import TaskLevelExportParameters
from otx.core.types.task import OTXTaskType


@pytest.fixture()
def fxt_tv_model():
    return TVModelForMulticlassCls(backbone="mobilenet_v3_small", label_info=10)


@pytest.fixture()
def fxt_tv_model_and_data_entity(
    request,
    fxt_multiclass_cls_batch_data_entity,
    fxt_multilabel_cls_batch_data_entity,
    fxt_hlabel_cls_batch_data_entity,
    fxt_hlabel_multilabel_info,
):
    if request.param == OTXTaskType.MULTI_CLASS_CLS:
        return TVModelForMulticlassCls(
            backbone="mobilenet_v3_small",
            label_info=10,
        ), fxt_multiclass_cls_batch_data_entity
    if request.param == OTXTaskType.MULTI_LABEL_CLS:
        return TVModelForMultilabelCls(
            backbone="mobilenet_v3_small",
            label_info=10,
        ), fxt_multilabel_cls_batch_data_entity
    if request.param == OTXTaskType.H_LABEL_CLS:
        return TVModelForHLabelCls(
            backbone="mobilenet_v3_small",
            label_info=fxt_hlabel_multilabel_info,
        ), fxt_hlabel_cls_batch_data_entity
    return None


class TestOTXTVModel:
    def test_create_model(self, fxt_tv_model):
        assert isinstance(fxt_tv_model.model, ImageClassifier)

        semi_sl_model = TVModelForMulticlassCls(
            backbone="mobilenet_v3_small",
            label_info=10,
            train_type="SEMI_SUPERVISED",
        )
        assert isinstance(semi_sl_model.model.head, SemiSLLinearClsHead)

    @pytest.mark.parametrize(
        "fxt_tv_model_and_data_entity",
        [OTXTaskType.MULTI_CLASS_CLS, OTXTaskType.MULTI_LABEL_CLS, OTXTaskType.H_LABEL_CLS],
        indirect=True,
    )
    def test_customize_inputs(self, fxt_tv_model_and_data_entity):
        tv_model, data_entity = fxt_tv_model_and_data_entity
        outputs = tv_model._customize_inputs(data_entity)
        assert "images" in outputs
        assert "labels" in outputs
        assert "mode" in outputs

    @pytest.mark.parametrize(
        "fxt_tv_model_and_data_entity",
        [OTXTaskType.MULTI_CLASS_CLS, OTXTaskType.MULTI_LABEL_CLS, OTXTaskType.H_LABEL_CLS],
        indirect=True,
    )
    def test_customize_outputs(self, fxt_tv_model_and_data_entity):
        tv_model, data_entity = fxt_tv_model_and_data_entity
        outputs = torch.randn(2, 10)
        tv_model.training = True
        preds = tv_model._customize_outputs(outputs, data_entity)
        assert isinstance(preds, OTXBatchLossEntity)

        tv_model.training = False
        preds = tv_model._customize_outputs(outputs, data_entity)
        assert isinstance(preds, OTXBatchPredEntity)

    def test_export_parameters(self, fxt_tv_model):
        export_parameters = fxt_tv_model._export_parameters
        assert isinstance(export_parameters, TaskLevelExportParameters)
        assert export_parameters.model_type == "Classification"
        assert export_parameters.task_type == "classification"

    @pytest.mark.parametrize("explain_mode", [True, False])
    def test_predict_step(self, fxt_tv_model, fxt_multiclass_cls_batch_data_entity, explain_mode):
        fxt_tv_model.eval()
        fxt_tv_model.explain_mode = explain_mode
        outputs = fxt_tv_model.predict_step(batch=fxt_multiclass_cls_batch_data_entity, batch_idx=0)

        assert isinstance(outputs, MulticlassClsBatchPredEntity)
        assert outputs.has_xai_outputs == explain_mode
        if explain_mode:
            assert outputs.feature_vector.ndim == 2
            assert outputs.saliency_map.ndim == 4
            assert outputs.saliency_map.shape[-2:] != torch.Size([1, 1])
