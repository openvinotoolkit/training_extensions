# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os

import numpy as np
import pytest
import torch
from mmdet.models import build_detector

from otx.algorithms.common.adapters.mmcv.utils.config_utils import OTXConfig
from otx.algorithms.detection.adapters.mmdet.hooks import DetClassProbabilityMapHook
from otx.algorithms.detection.adapters.mmdet.hooks.det_class_probability_map_hook import MaskRCNNRecordingForwardHook
from otx.cli.registry import Registry
from tests.test_suite.e2e_test_system import e2e_pytest_unit

templates_det = Registry("src/otx/algorithms").filter(task_type="DETECTION").templates
templates_det_ids = [template.model_template_id for template in templates_det]

templates_two_stage_det = Registry("src/otx/algorithms/detection").filter(task_type="INSTANCE_SEGMENTATION").templates
templates_two_stage_det_ids = [template.model_template_id for template in templates_two_stage_det]


class TestExplainMethods:
    ref_saliency_shapes = {
        "MobileNetV2-ATSS": (2, 4, 4),
        "ResNeXt101-ATSS": (2, 4, 4),
        "SSD": (81, 13, 13),
        "YOLOX-TINY": (80, 13, 13),
        "YOLOX-S": (80, 13, 13),
        "YOLOX-L": (80, 13, 13),
        "YOLOX-X": (80, 13, 13),
    }

    ref_saliency_vals_det = {
        "MobileNetV2-ATSS": np.array([67, 216, 255, 57], dtype=np.uint8),
        "ResNeXt101-ATSS": np.array([75, 214, 229, 173], dtype=np.uint8),
        "YOLOX-TINY": np.array([80, 28, 42, 53, 49, 68, 72, 75, 69, 57, 65, 6, 157], dtype=np.uint8),
        "YOLOX-S": np.array([75, 178, 151, 159, 150, 148, 144, 144, 147, 144, 147, 142, 189], dtype=np.uint8),
        "YOLOX-L": np.array([43, 28, 0, 6, 7, 19, 22, 17, 14, 18, 25, 7, 34], dtype=np.uint8),
        "YOLOX-X": np.array([255, 144, 83, 76, 83, 86, 82, 90, 91, 93, 110, 104, 83], dtype=np.uint8),
        "SSD": np.array([119, 72, 118, 35, 39, 30, 31, 31, 36, 27, 44, 23, 61], dtype=np.uint8),
    }

    ref_saliency_vals_det_wo_postprocess = {
        "MobileNetV2-ATSS": -0.10465062,
        "ResNeXt101-ATSS": -0.073549636,
        "YOLOX-TINY": 0.04948914,
        "YOLOX-S": 0.01133332,
        "YOLOX-L": 0.01870133,
        "YOLOX-X": 0.0043506604,
        "SSD": 0.6629989,
    }

    @staticmethod
    def _get_model(template):
        torch.manual_seed(0)

        base_dir = os.path.abspath(os.path.dirname(template.model_template_path))
        cfg_path = os.path.join(base_dir, "model.py")
        cfg = OTXConfig.fromfile(cfg_path)

        model = build_detector(cfg.model)
        model = model.eval()
        return model

    @staticmethod
    def _get_data():
        img = torch.ones(2, 3, 416, 416) - 0.5
        img_metas = [
            {
                "img_shape": (416, 416, 3),
                "ori_shape": (416, 416, 3),
                "scale_factor": np.array([1.1784703, 0.832, 1.1784703, 0.832], dtype=np.float32),
            },
        ] * 2
        data = {"img_metas": [img_metas], "img": [img]}
        return data

    @e2e_pytest_unit
    @pytest.mark.parametrize("template", templates_det, ids=templates_det_ids)
    def test_saliency_map_det(self, template):
        model = self._get_model(template)
        data = self._get_data()

        with DetClassProbabilityMapHook(model) as det_hook:
            with torch.no_grad():
                _ = model(return_loss=False, rescale=True, **data)
        saliency_maps = det_hook.records

        assert len(saliency_maps) == 2
        assert saliency_maps[0].ndim == 3
        assert saliency_maps[0].shape == self.ref_saliency_shapes[template.name]
        # convert to int16 in case of negative value difference
        actual_sal_vals = saliency_maps[0][0][0].astype(np.int16)
        ref_sal_vals = self.ref_saliency_vals_det[template.name].astype(np.uint8)
        assert np.all(np.abs(actual_sal_vals - ref_sal_vals) <= 1)

    @e2e_pytest_unit
    @pytest.mark.parametrize("template", templates_det, ids=templates_det_ids)
    def test_saliency_map_det_wo_postprocessing(self, template):
        model = self._get_model(template)
        data = self._get_data()

        with DetClassProbabilityMapHook(model, normalize=False, use_cls_softmax=False) as det_hook:
            with torch.no_grad():
                _ = model(return_loss=False, rescale=True, **data)
        saliency_maps = det_hook.records

        assert len(saliency_maps) == 2
        assert saliency_maps[0].ndim == 3
        assert saliency_maps[0].shape == self.ref_saliency_shapes[template.name]
        assert np.abs(saliency_maps[0][0][0][0] - self.ref_saliency_vals_det_wo_postprocess[template.name]) < 1e-4

    @e2e_pytest_unit
    @pytest.mark.parametrize("template", templates_two_stage_det, ids=templates_two_stage_det_ids)
    def test_saliency_map_two_stage_det(self, template):
        model = self._get_model(template)
        data = self._get_data()

        with MaskRCNNRecordingForwardHook(model, input_img_shape=(800, 1344)) as det_hook:
            with torch.no_grad():
                _ = model(return_loss=False, rescale=True, **data)
        saliency_maps = det_hook.records

        # MaskRCNNRecordingForwardHook generates saliency maps based on predictions.
        # Current test does not intend to test a trained model - so no prediction and no saliency maps are available.
        assert saliency_maps == [[None] * model.roi_head.mask_head.num_classes] * 2
