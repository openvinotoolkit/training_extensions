# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import torch
from otx.algo.classification.deit_tiny import (
    DeitTinyForHLabelCls,
    DeitTinyForMulticlassCls,
    DeitTinyForMultilabelCls,
)
from otx.algo.utils.support_otx_v1 import OTXv1Helper


class TestDeitTiny:
    @pytest.mark.parametrize(
        "model_cls",
        [DeitTinyForMulticlassCls, DeitTinyForMultilabelCls, DeitTinyForHLabelCls],
    )
    def test_deit_tiny(self, model_cls, mocker, fxt_hlabel_data):
        num_classes = fxt_hlabel_data.num_classes

        if model_cls == DeitTinyForHLabelCls:
            model = model_cls(hlabel_info=fxt_hlabel_data)
        else:
            model = model_cls(num_classes=num_classes)
        model.model.explain_fn = model.get_explain_fn()

        assert model._optimization_config["model_type"] == "transformer"

        assert model.head_forward_fn(torch.randn([1, 24, 192])).shape == torch.Size([1, num_classes])

        out = model._forward_explain_image_classifier(model.model, torch.randn(1, 3, 24, 24))
        assert out["logits"].shape == torch.Size([1, num_classes])
        assert out["feature_vector"].shape == torch.Size([1, 192])
        assert out["saliency_map"].shape == torch.Size([1, num_classes, 2, 2])

        mock_load_ckpt = mocker.patch.object(OTXv1Helper, "load_cls_effnet_b0_ckpt")
        model.load_from_otx_v1_ckpt({})
        mock_load_ckpt.assert_called_once_with({}, "multiclass", "model.model.")
