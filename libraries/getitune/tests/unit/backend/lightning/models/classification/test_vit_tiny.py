# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest

from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.lightning.models.classification.multiclass_models.vit import VisionTransformerMulticlassCls
from getitune.backend.lightning.models.classification.multilabel_models.vit import VisionTransformerMultilabelCls
from getitune.data.entity.base import BatchLoss


class TestVitTiny:
    @pytest.fixture(
        params=[
            (VisionTransformerMulticlassCls, "fxt_multiclass_cls_batch_data_entity", "fxt_multiclass_labelinfo"),
            (VisionTransformerMultilabelCls, "fxt_multilabel_cls_batch_data_entity", "fxt_multilabel_labelinfo"),
        ],
        ids=["multiclass", "multilabel"],
    )
    def fxt_model_and_input(self, request):
        model_cls, input_fxt_name, label_info_fxt_name = request.param
        fxt_input = request.getfixturevalue(input_fxt_name)
        fxt_label_info = request.getfixturevalue(label_info_fxt_name)

        model = model_cls(
            label_info=fxt_label_info,
            data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )

        return model, fxt_input

    @pytest.mark.parametrize("explain_mode", [True, False])
    def test_vit_tiny(self, fxt_model_and_input, explain_mode, mocker):
        fxt_model, fxt_input = fxt_model_and_input

        fxt_model.train()
        assert isinstance(fxt_model(fxt_input), BatchLoss)

        fxt_model.eval()
        assert not isinstance(fxt_model(fxt_input), BatchLoss)

        fxt_model.explain_mode = explain_mode
        preds = fxt_model.predict_step(fxt_input, batch_idx=0)
        assert len(preds.labels) == fxt_input.batch_size
        assert len(preds.scores) == fxt_input.batch_size
        assert (preds.saliency_map is not None and len(preds.saliency_map) > 0) == explain_mode

    @pytest.mark.parametrize(
        ("model_cls", "label_info_fxt_name"),
        [
            (VisionTransformerMulticlassCls, "fxt_multiclass_labelinfo"),
            (VisionTransformerMultilabelCls, "fxt_multilabel_labelinfo"),
        ],
        ids=["multiclass", "multilabel"],
    )
    def test_freeze_backbone(self, model_cls, label_info_fxt_name, request):
        data_input_params = DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        fxt_label_info = request.getfixturevalue(label_info_fxt_name)

        # Test with freeze_backbone=True
        model = model_cls(
            label_info=fxt_label_info,
            data_input_params=data_input_params,
            freeze_backbone=True,
        )

        classification_layers = model._identify_classification_layers()
        assert all(param.requires_grad == (name in classification_layers) for name, param in model.named_parameters())

        # Test with freeze_backbone=False
        model = model_cls(
            label_info=fxt_label_info,
            data_input_params=data_input_params,
            freeze_backbone=False,
        )

        assert all(param.requires_grad for param in model.parameters())
