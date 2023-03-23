# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os

import numpy as np
import pytest
import torch
from mmcls.models import build_classifier
from mmdet.models import build_detector

from otx.algorithms.classification.tasks import ClassificationInferenceTask  # noqa
from otx.algorithms.common.adapters.mmcv.hooks.recording_forward_hook import (
    ReciproCAMHook,
)
from otx.algorithms.common.adapters.mmcv.utils.config_utils import MPAConfig
from otx.algorithms.detection.adapters.mmdet.hooks import DetSaliencyMapHook
from otx.cli.registry import Registry
from otx.mpa.det.stage import DetectionStage  # noqa
from tests.test_suite.e2e_test_system import e2e_pytest_unit

templates_cls = Registry("otx/algorithms").filter(task_type="CLASSIFICATION").templates
templates_cls_ids = [template.model_template_id for template in templates_cls]

templates_det = Registry("otx/algorithms").filter(task_type="DETECTION").templates
templates_det_ids = [template.model_template_id for template in templates_det]


class TestExplainMethods:
    ref_saliency_vals_cls = {
        "EfficientNet-B0": np.array([153, 102, 188, 189, 199, 159, 170], dtype=np.uint8),
        "MobileNet-V3-large-1x": np.array([140, 82, 87, 81, 79, 117, 254], dtype=np.uint8),
        "EfficientNet-V2-S": np.array([92, 197, 220, 227, 222, 196, 77], dtype=np.uint8),
    }

    ref_saliency_shapes = {
        "ATSS": (2, 4, 4),
        "SSD": (81, 13, 13),
        "YOLOX": (80, 13, 13),
    }

    ref_saliency_vals_det = {
        "ATSS": np.array([80, 213, 255, 124], dtype=np.uint8),
        "YOLOX": np.array([99, 38, 42, 57, 50, 71, 75, 81, 75, 61, 70, 5, 177], dtype=np.uint8),
        "SSD": np.array([191, 133, 160, 60, 63, 50, 51, 51, 57, 54, 83, 71, 148], dtype=np.uint8),
    }

    @e2e_pytest_unit
    @pytest.mark.parametrize("template", templates_cls, ids=templates_cls_ids)
    def test_saliency_map_cls(self, template):
        torch.manual_seed(0)

        base_dir = os.path.abspath(os.path.dirname(template.model_template_path))
        cfg_path = os.path.join(base_dir, "model.py")
        cfg = MPAConfig.fromfile(cfg_path)

        cfg.model.pop("task")
        model = build_classifier(cfg.model)
        model = model.eval()

        # img = torch.rand(2, 3, 224, 224) - 0.5
        img = torch.ones(2, 3, 224, 224) - 0.5
        data = {"img_metas": {}, "img": img}

        with ReciproCAMHook(model) as rcam_hook:
            with torch.no_grad():
                _ = model(return_loss=False, **data)
        saliency_maps = rcam_hook.records

        assert len(saliency_maps) == 2
        assert saliency_maps[0].ndim == 3
        assert saliency_maps[0].shape == (1000, 7, 7)
        assert (saliency_maps[0][0][0] == self.ref_saliency_vals_cls[template.name]).all()

    @e2e_pytest_unit
    @pytest.mark.parametrize("template", templates_det, ids=templates_det_ids)
    def test_saliency_map_det(self, template):
        torch.manual_seed(0)

        base_dir = os.path.abspath(os.path.dirname(template.model_template_path))
        cfg_path = os.path.join(base_dir, "model.py")
        cfg = MPAConfig.fromfile(cfg_path)

        model = build_detector(cfg.model)
        model = model.eval()

        # img = torch.rand(2, 3, 416, 416) - 0.5
        img = torch.ones(2, 3, 416, 416) - 0.5
        img_metas = [
            {
                "img_shape": (416, 416, 3),
                "scale_factor": np.array([1.1784703, 0.832, 1.1784703, 0.832], dtype=np.float32),
            },
            {
                "img_shape": (416, 416, 3),
                "scale_factor": np.array([1.1784703, 0.832, 1.1784703, 0.832], dtype=np.float32),
            },
        ]
        data = {"img_metas": [img_metas], "img": [img]}

        with DetSaliencyMapHook(model) as det_hook:
            with torch.no_grad():
                _ = model(return_loss=False, rescale=True, **data)
        saliency_maps = det_hook.records

        assert len(saliency_maps) == 2
        assert saliency_maps[0].ndim == 3
        assert saliency_maps[0].shape == self.ref_saliency_shapes[template.name]
        assert (saliency_maps[0][0][0] == self.ref_saliency_vals_det[template.name]).all()
