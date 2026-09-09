# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of getitune YOLOX architecture."""

import pytest
import torch

from getitune.backend.lightning.exporter.native import LightningModelExporter
from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.lightning.models.detection.backbones.csp_darknet import CSPDarknetModule
from getitune.backend.lightning.models.detection.heads.yolox_head import YOLOXHeadModule
from getitune.backend.lightning.models.detection.necks.yolox_pafpn import YOLOXPAFPNModule
from getitune.backend.lightning.models.detection.yolox import YOLOX
from getitune.data.entity.sample import PredictionBatch


class TestYOLOX:
    @pytest.fixture(params=["yolox_tiny"])
    def fxt_model(self, request) -> YOLOX:
        return YOLOX(
            model_name=request.param,
            label_info=3,
            data_input_params=DataInputParams((320, 320), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pretrained=False,
        )

    def test_init(self) -> None:
        yolox_l = YOLOX(
            model_name="yolox_l",
            label_info=3,
            data_input_params=DataInputParams((320, 320), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pretrained=False,
        )
        assert isinstance(yolox_l.model.backbone, CSPDarknetModule)
        assert isinstance(yolox_l.model.neck, YOLOXPAFPNModule)
        assert isinstance(yolox_l.model.bbox_head, YOLOXHeadModule)
        assert yolox_l.data_input_params.input_size == (320, 320)

        yolox_tiny = YOLOX(
            model_name="yolox_tiny",
            label_info=3,
            data_input_params=DataInputParams((320, 320), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pretrained=False,
        )
        assert yolox_tiny.data_input_params.input_size == (320, 320)

        yolox_tiny = YOLOX(
            model_name="yolox_tiny",
            label_info=3,
            data_input_params=DataInputParams((416, 416), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pretrained=False,
        )
        assert yolox_tiny.data_input_params.input_size == (416, 416)

    def test_exporter(self) -> None:
        yolox_l = YOLOX(
            model_name="yolox_l",
            label_info=3,
            data_input_params=DataInputParams((320, 320), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pretrained=False,
        )
        yolox_l_exporter = yolox_l._exporter
        assert isinstance(yolox_l_exporter, LightningModelExporter)
        assert yolox_l_exporter.swap_rgb is True

        yolox_tiny = YOLOX(
            model_name="yolox_tiny",
            label_info=3,
            data_input_params=DataInputParams((320, 320), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pretrained=False,
        )
        yolox_tiny_exporter = yolox_tiny._exporter
        assert isinstance(yolox_tiny_exporter, LightningModelExporter)
        assert yolox_tiny_exporter.swap_rgb is False

    def test_loss(self, fxt_model, fxt_detection_batch):
        output = fxt_model(fxt_detection_batch)
        assert "loss_cls" in output
        assert "loss_bbox" in output
        assert "loss_obj" in output

    def test_predict(self, fxt_model, fxt_detection_batch):
        fxt_model.eval()
        output = fxt_model(fxt_detection_batch)
        assert isinstance(output, PredictionBatch)

    def test_export(self, fxt_model):
        fxt_model.eval()
        output = fxt_model.forward_for_tracing(torch.randn(1, 3, 32, 32))
        assert len(output) == 2

        fxt_model.explain_mode = True
        output = fxt_model.forward_for_tracing(torch.randn(1, 3, 32, 32))
        assert len(output) == 4

    def test_export_nms_disabled(self, fxt_model):
        fxt_model.eval()
        fxt_model.export_nms = False
        dets, labels = fxt_model.forward_for_tracing(torch.randn(1, 3, 32, 32))
        assert dets.ndim == 3
        assert dets.shape[0] == 1
        assert dets.shape[2] == 5
        assert labels.shape == dets.shape[:2]

    @pytest.mark.parametrize("model_name", ["yolox_s", "yolox_l", "yolox_x", "yolox_tiny"])
    def test_no_intensity_config_in_defaults(self, model_name):
        """Default DataInputParams for all YOLOX variants have no intensity_config.

        At runtime the engine propagates IntensityConfig from SubsetConfig (always
        present) so export metadata is correct without hardcoding it in defaults.
        """
        # Use default params (no explicit data_input_params)
        model = YOLOX(model_name=model_name, label_info=3, pretrained=False)
        assert model.data_input_params.intensity_config is None

    @pytest.mark.parametrize("model_name", ["yolox_s", "yolox_l", "yolox_x"])
    @pytest.mark.parametrize("storage_dtype", ["uint16", "int16"])
    def test_raw_uint8_models_reject_high_bit_depth(self, model_name, storage_dtype):
        from getitune.config.data import IntensityConfig

        intensity_cfg = IntensityConfig(storage_dtype=storage_dtype, mode="scale_to_unit")
        params = DataInputParams((320, 320), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), intensity_config=intensity_cfg)
        with pytest.raises(ValueError, match="does not support high-bit-depth"):
            YOLOX(model_name=model_name, label_info=3, data_input_params=params, pretrained=False)
