# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os

import numpy as np
import pytest
import torch
from mmcls.models import build_classifier

from otx.algorithms.classification.adapters.mmcls.configurer import ClassificationConfigurer
from otx.algorithms.common.adapters.mmcv.hooks.recording_forward_hook import (
    ReciproCAMHook,
)
from otx.algorithms.common.adapters.mmcv.utils.config_utils import MPAConfig
from otx.cli.registry import Registry
from tests.test_suite.e2e_test_system import e2e_pytest_unit

templates_cls = Registry("src/otx/algorithms").filter(task_type="CLASSIFICATION").templates
templates_cls_ids = [template.model_template_id for template in templates_cls]


class TestExplainMethods:
    ref_saliency_vals_cls = {
        "EfficientNet-B0": np.array([57, 0, 161, 127, 102, 96, 92], dtype=np.uint8),
        "MobileNet-V3-large-1x": np.array([140, 82, 87, 81, 79, 117, 254], dtype=np.uint8),
        "EfficientNet-V2-S": np.array([125, 42, 24, 21, 27, 55, 145], dtype=np.uint8),
    }

    @e2e_pytest_unit
    @pytest.mark.parametrize("template", templates_cls, ids=templates_cls_ids)
    def test_saliency_map_cls(self, template):
        if template.name == "DeiT-Tiny":
            pytest.skip(reason="Issue#2098 ViT inference does not work by FeatureVectorHook.")
        torch.manual_seed(0)
        base_dir = os.path.abspath(os.path.dirname(template.model_template_path))
        cfg_path = os.path.join(base_dir, "model.py")
        cfg = MPAConfig.fromfile(cfg_path)

        cfg.model.pop("task")
        ClassificationConfigurer.configure_in_channel(cfg)
        model = build_classifier(cfg.model)
        model = model.eval()

        img = torch.ones(2, 3, 224, 224) - 0.5
        data = {"img_metas": {}, "img": img}

        with ReciproCAMHook(model) as rcam_hook:
            with torch.no_grad():
                _ = model(return_loss=False, **data)
        saliency_maps = rcam_hook.records

        assert len(saliency_maps) == 2
        assert saliency_maps[0].ndim == 3
        assert saliency_maps[0].shape == (1000, 7, 7)
        actual_sal_vals = saliency_maps[0][0][0].astype(np.int8)
        ref_sal_vals = self.ref_saliency_vals_cls[template.name].astype(np.int8)
        assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)
