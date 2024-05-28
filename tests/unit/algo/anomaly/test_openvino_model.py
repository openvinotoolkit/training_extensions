# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from anomalib.metrics.min_max import MinMax
from anomalib.metrics.threshold import ManualThreshold
from otx.algo.anomaly.openvino_model import AnomalyOpenVINO
from otx.algo.anomaly.padim import Padim
from otx.algo.anomaly.stfpm import Stfpm
from otx.core.types.export import OTXExportFormatType
from otx.core.types.label import AnomalyLabelInfo
from otx.core.types.task import OTXTaskType


class TestAnomalyOpenVINO:
    @pytest.fixture(
        params=[
            # "padim", Cannot export padim in this way. We will face an error.
            "stfpm",
        ],
    )
    def fxt_ov_ir(self, request, tmpdir):
        if request.param == "padim":
            model = Padim()
        elif request.param == "stfpm":
            model = Stfpm()
        else:
            raise ValueError

        # NOTE: if we do not inject those into `model`,
        # we will face errors during `model.export()` such as:
        # AttributeError: 'Stfpm' object has no attribute 'normalization_metrics'
        model.normalization_metrics = MinMax()
        model.normalization_metrics.min = torch.tensor(-1.0)
        model.normalization_metrics.max = torch.tensor(1.0)
        model.image_threshold = ManualThreshold(0.0)
        model.pixel_threshold = ManualThreshold(0.0)
        model.task = OTXTaskType.ANOMALY_CLASSIFICATION

        return model.export(
            output_dir=tmpdir,
            base_name="exported_model",
            export_format=OTXExportFormatType.OPENVINO,
        )

    def test_label_info(self, fxt_ov_ir):
        ov_model = AnomalyOpenVINO(model_name=fxt_ov_ir)
        assert isinstance(ov_model.label_info, AnomalyLabelInfo)
