# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from otx.algo.classification.efficientnet_b0 import (
    EfficientNetB0ForHLabelCls,
    EfficientNetB0ForMulticlassCls,
    EfficientNetB0ForMultilabelCls,
)
from otx.algo.classification.efficientnet_v2 import (
    EfficientNetV2ForHLabelCls,
    EfficientNetV2ForMulticlassCls,
    EfficientNetV2ForMultilabelCls,
)
from otx.algo.utils.support_otx_v1 import OTXv1Helper


class TestEfficientB0:
    def test_effnet_b0_multiclass(self, mocker):
        mock_load_ckpt = mocker.patch.object(OTXv1Helper, "load_cls_effnet_b0_ckpt")

        model = EfficientNetB0ForMulticlassCls(num_classes=10)
        model.load_from_otx_v1_ckpt({})
        mock_load_ckpt.assert_called_once_with({}, "multiclass", "model.model.")

    def test_effnet_b0_multilabel(self, mocker):
        mock_load_ckpt = mocker.patch.object(OTXv1Helper, "load_cls_effnet_b0_ckpt")
        model = EfficientNetB0ForMultilabelCls(num_classes=10)
        model.load_from_otx_v1_ckpt({})
        mock_load_ckpt.assert_called_once_with({}, "multilabel", "model.model.")

    def test_effnet_b0_hlabel(self, mocker):
        mock_load_ckpt = mocker.patch.object(OTXv1Helper, "load_cls_effnet_b0_ckpt")
        model = EfficientNetB0ForHLabelCls(num_classes=10, num_multiclass_heads=2, num_multilabel_classes=5)
        model.load_from_otx_v1_ckpt({})
        mock_load_ckpt.assert_called_once_with({}, "hlabel", "model.model.")


class TestEfficientV2:
    def test_effnet_v2_multiclass(self, mocker):
        mock_load_ckpt = mocker.patch.object(OTXv1Helper, "load_cls_effnet_v2_ckpt")

        model = EfficientNetV2ForMulticlassCls(num_classes=10)
        model.load_from_otx_v1_ckpt({})
        mock_load_ckpt.assert_called_once_with({}, "multiclass", "model.model.")

    def test_effnet_v2_multilabel(self, mocker):
        mock_load_ckpt = mocker.patch.object(OTXv1Helper, "load_cls_effnet_v2_ckpt")

        model = EfficientNetV2ForMultilabelCls(num_classes=10)
        model.load_from_otx_v1_ckpt({})
        mock_load_ckpt.assert_called_once_with({}, "multilabel", "model.model.")

    def test_effnet_v2_hlabel(self, mocker):
        mock_load_ckpt = mocker.patch.object(OTXv1Helper, "load_cls_effnet_v2_ckpt")

        model = EfficientNetV2ForHLabelCls(num_classes=10, num_multiclass_heads=2, num_multilabel_classes=5)
        model.load_from_otx_v1_ckpt({})
        mock_load_ckpt.assert_called_once_with({}, "hlabel", "model.model.")
