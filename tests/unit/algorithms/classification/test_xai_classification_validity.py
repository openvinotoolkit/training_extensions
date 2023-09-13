# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os

import numpy as np
import pytest
import torch
from torch.nn import LayerNorm
from mmcls.models import build_classifier

from otx.algorithms.classification.adapters.mmcls.configurer import ClassificationConfigurer
from otx.algorithms.common.adapters.mmcv.hooks.recording_forward_hook import (
    ReciproCAMHook,
    ViTReciproCAMHook,
)
from otx.algorithms.common.adapters.mmcv.utils.config_utils import OTXConfig
from otx.cli.registry import Registry
from otx.algorithms.classification.adapters.mmcls.models.classifiers.custom_image_classifier import _extract_vit_feat
from tests.test_suite.e2e_test_system import e2e_pytest_unit

templates_cls = Registry("src/otx/algorithms").filter(task_type="CLASSIFICATION").templates
templates_cls_ids = [template.model_template_id for template in templates_cls]


class TestExplainMethods:
    ref_saliency_vals_cls = {
        "EfficientNet-B0": np.array([57, 0, 161, 127, 102, 96, 92], dtype=np.uint8),
        "MobileNet-V3-large-1x": np.array([140, 82, 87, 81, 79, 117, 254], dtype=np.uint8),
        "EfficientNet-V2-S": np.array([125, 42, 24, 21, 27, 55, 145], dtype=np.uint8),
        "DeiT-Tiny": np.array([0, 108, 108, 108, 108, 108, 108, 108, 108, 109, 109, 109, 109, 0], dtype=np.uint8),
    }

    @e2e_pytest_unit
    @pytest.mark.parametrize("template", templates_cls, ids=templates_cls_ids)
    def test_saliency_map_cls(self, template):
        torch.manual_seed(0)
        base_dir = os.path.abspath(os.path.dirname(template.model_template_path))
        cfg_path = os.path.join(base_dir, "model.py")
        cfg = OTXConfig.fromfile(cfg_path)

        cfg.model.pop("task")
        ClassificationConfigurer.configure_in_channel(cfg)
        model = build_classifier(cfg.model)
        model = model.eval()

        img = torch.ones(2, 3, 224, 224) - 0.5
        data = {"img_metas": {}, "img": img}

        if template.name == "DeiT-Tiny":
            explainer_hook = ViTReciproCAMHook
            saliency_map_ref_shape = (1000, 14, 14)
        else:
            explainer_hook = ReciproCAMHook
            saliency_map_ref_shape = (1000, 7, 7)

        with explainer_hook(model) as rcam_hook:
            with torch.no_grad():
                _ = model(return_loss=False, **data)
        saliency_maps = rcam_hook.records

        assert len(saliency_maps) == 2
        assert saliency_maps[0].ndim == 3
        assert saliency_maps[0].shape == saliency_map_ref_shape
        # convert to int16 in case of negative value difference
        actual_sal_vals = saliency_maps[0][0][0].astype(np.int16)
        ref_sal_vals = self.ref_saliency_vals_cls[template.name].astype(np.uint8)
        assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)


class TestViTExplain:
    DEIT_TEMPLATE_DIR = os.path.join("src/otx/algorithms/classification/configs", "deit_tiny")

    def _create_model(self):
        torch.manual_seed(0)
        base_dir = os.path.abspath(self.DEIT_TEMPLATE_DIR)
        cfg_path = os.path.join(base_dir, "model.py")
        cfg = OTXConfig.fromfile(cfg_path)

        cfg.model.pop("task")
        ClassificationConfigurer.configure_in_channel(cfg)
        model = build_classifier(cfg.model)
        model = model.eval()
        return model

    @e2e_pytest_unit
    def test_extract_vit_feat(self):
        model = self._create_model()

        img = torch.ones(2, 3, 224, 224) - 0.5
        feat, layernorm_feat = _extract_vit_feat(model, img)

        assert len(feat) == 2
        assert feat[0].shape == torch.Size([2, 192, 14, 14])
        assert feat[1].shape == torch.Size([2, 192])
        assert abs(feat[0][0][0][0][0].detach().cpu().item() - 0.4621) < 0.05
        assert layernorm_feat.shape == torch.Size([2, 197, 192])
        assert abs(layernorm_feat[0][0][0].detach().cpu().item() - 0.7244) < 0.05

    @e2e_pytest_unit
    @pytest.mark.parametrize("layer_index", [-1, -2])
    @pytest.mark.parametrize("use_gaussian", [True, False])
    @pytest.mark.parametrize("cls_token", [True, False])
    def test_vit_reciprocam_hook_initiate(self, layer_index, use_gaussian, cls_token):
        model = self._create_model()

        explainer_hook = ViTReciproCAMHook(model, layer_index, use_gaussian, cls_token)
        assert explainer_hook.records == []
        assert isinstance(explainer_hook._target_layernorm, LayerNorm)

        mosaic_feature_map = explainer_hook._get_mosaic_feature_map(torch.ones(197, 192))
        logit = explainer_hook._predict_from_feature_map(torch.ones(2, 197, 192))
        assert mosaic_feature_map is not None
        assert logit is not None
        assert mosaic_feature_map.shape == torch.Size([196, 197, 192])
        assert logit.shape == torch.Size([2, 1000])

    @e2e_pytest_unit
    @pytest.mark.parametrize("layer_index", [-1, -2])
    @pytest.mark.parametrize("use_gaussian", [True, False])
    @pytest.mark.parametrize("cls_token", [True, False])
    def test_vit_reciprocam_hook_func(self, layer_index, use_gaussian, cls_token):
        model = self._create_model()

        explainer_hook = ViTReciproCAMHook(model, layer_index, use_gaussian, cls_token)
        img = torch.ones(2, 3, 224, 224) - 0.5
        _, layernorm_feat = _extract_vit_feat(model, img)
        saliency_map = explainer_hook.func(layernorm_feat)
        assert saliency_map is not None
