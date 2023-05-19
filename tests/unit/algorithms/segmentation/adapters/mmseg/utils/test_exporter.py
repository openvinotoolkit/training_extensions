# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os

from otx.algorithms.segmentation.adapters.mmseg.utils.builder import build_segmentor
from otx.algorithms.common.adapters.mmdeploy.apis import NaiveExporter
from otx.algorithms.common.adapters.mmcv.tasks.exporter import Exporter
from otx.algorithms.segmentation.adapters.mmseg.utils.exporter import SegmentationExporter
from otx.algorithms.common.adapters.mmcv.utils.config_utils import MPAConfig
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.segmentation.test_helpers import (
    DEFAULT_SEG_TEMPLATE_DIR,
)


@e2e_pytest_unit
def test_run(mocker):
    exporter = SegmentationExporter()
    model_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_SEG_TEMPLATE_DIR, "model.py"))
    model_cfg.work_dir = "/tmp/"
    args = {"precision": "FP32", "model_builder": build_segmentor}
    mocker.patch.object(Exporter, "run", return_value=True)
    returned_value = exporter.run(model_cfg)
    assert "model_builder" in args
    assert returned_value is True


@e2e_pytest_unit
def test_naive_export(mocker):
    exporter = SegmentationExporter()
    data_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_SEG_TEMPLATE_DIR, "data_pipeline.py"))
    mock_export_ov = mocker.patch.object(NaiveExporter, "export2backend")
    exporter.naive_export("", build_segmentor, "FP32", "OPENVINO", data_cfg)
    mock_export_ov.assert_called_once()
