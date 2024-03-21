# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from otx.algo.classification.mobilenet_v3_large import (
    MobileNetV3ForHLabelCls,
    MobileNetV3ForMulticlassCls,
    MobileNetV3ForMultilabelCls,
)

from otx.algo.utils.support_otx_v1 import OTXv1Helper

class TestMobileNetV3:
    def test_mobilenet_v3_multiclass(self, mocker):
        mock_load_ckpt = mocker.patch.object(OTXv1Helper, "load_cls_mobilenet_v3_ckpt")
        
        model = MobileNetV3ForMulticlassCls(num_classes=10)
        model.load_from_otx_v1_ckpt({})
        mock_load_ckpt.assert_called_once_with({}, "multiclass", "model.model.")
        assert isinstance(model._export_parameters, dict)
        
    def test_mobilenet_v3_multilabel(self, mocker):
        mock_load_ckpt = mocker.patch.object(OTXv1Helper, "load_cls_mobilenet_v3_ckpt")
        model = MobileNetV3ForMultilabelCls(num_classes=10)
        model.load_from_otx_v1_ckpt({})
        mock_load_ckpt.assert_called_once_with({}, "multilabel", "model.model.")
        assert isinstance(model._export_parameters, dict)
        
    def test_mobilenet_v3_hlabel(self, mocker):
        mock_load_ckpt = mocker.patch.object(OTXv1Helper, "load_cls_mobilenet_v3_ckpt")
        model = MobileNetV3ForHLabelCls(num_classes=10, num_multiclass_heads=2, num_multilabel_classes=5)
        model.load_from_otx_v1_ckpt({})
        mock_load_ckpt.assert_called_once_with({}, "hlabel", "model.model.")