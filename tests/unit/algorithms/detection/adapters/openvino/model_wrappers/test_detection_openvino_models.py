"""Unit Test for otx.algorithms.detection.adapters.openvino.model_wrappers.openvino_models."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Dict

import numpy as np
import pytest
from mmcv.utils import Config
from openvino.model_api.adapters import OpenvinoAdapter

from otx.algorithms.detection.adapters.openvino.model_wrappers.openvino_models import (
    OTXMaskRCNNModel,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class MockOpenvinoAdapter(OpenvinoAdapter):
    """Mock class for OpenvinoAdapter."""

    def __init__(self):
        pass


class MockOTXMaskRCNNModel(OTXMaskRCNNModel):
    """Mock class for OTXMaskRCNNModel."""

    def __init__(self, *args):
        self.inputs: Dict[str, np.ndarray] = {
            "image": np.ndarray([1, 3, 10, 10]),
        }

        self.outputs: Dict[str, Config] = {
            "boxes": Config({"names": "boxes", "shape": [1, 1, 5]}),
            "labels": Config({"names": "labels", "shape": [1, 1]}),
            "masks": Config({"names": "masks", "shape": [1, 0, 28, 28]}),
            "feature_vector": Config({"names": "feature_vector", "shape": [1, 1, 1, 1]}),
            "saliency_map": Config({"names": "saliency_map", "shape": [1, 1, 1]}),
        }
        self.is_segmentoly = len(self.inputs) == 2
        self.output_blob_name = self._get_outputs()
        self.confidence_threshold = 0.5
        self.orig_width = 100
        self.orig_height = 100
        self.resize_type = ""
        super().__init__(MockOpenvinoAdapter, {})


class TestOTXMaskRCNNModel:
    """Test OTXMaskRCNNModel class.

    Test postprocess function
    <Steps>
        1. Generate sample output & meta
        2. Check whether postprocess function returns (scores, classes, boxes, resized_masks) tuple with length 4.
    """

    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        mocker.patch(
            "openvino.model_api.models.MaskRCNNModel.__init__",
            return_value=True,
        )
        self.model = MockOTXMaskRCNNModel()

    @e2e_pytest_unit
    def test_postprocess(self) -> None:
        """Test postprocess function."""

        sample_output = {
            "boxes": np.random.rand(1, 1, 5),
            "labels": np.random.rand(1, 1),
            "masks": np.random.rand(1, 1, 28, 28),
            "feature_vector": np.random.rand(1, 1, 1, 1),
            "saliency_map": np.random.rand(1, 1, 21),
        }
        sample_meta = {"original_shape": (10, 10, 3), "resized_shape": (5, 5, 3)}
        out = self.model.postprocess(sample_output, meta=sample_meta)
        assert len(out) == 4
