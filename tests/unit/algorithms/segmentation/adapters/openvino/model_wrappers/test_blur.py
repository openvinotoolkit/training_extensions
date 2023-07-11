# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest
from openvino.model_api.adapters.openvino_adapter import OpenvinoAdapter
from openvino.model_api.models import SegmentationModel

from otx.algorithms.segmentation.adapters.openvino.model_wrappers.blur import (
    BlurSegmentation,
    get_activation_map,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
def test_get_activation_map():
    fake_features = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    returned_value = get_activation_map(fake_features)

    assert type(returned_value).__module__ == np.__name__
    assert len(returned_value) == len(fake_features)


class TestBlurSegmentation:
    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        class MockAdapter(OpenvinoAdapter):
            def __init__(self):
                pass

        mocker.patch.object(SegmentationModel, "__init__")
        model_adapter = MockAdapter()
        self.blur = BlurSegmentation(model_adapter)

    @e2e_pytest_unit
    def test_parameters(self):
        params = BlurSegmentation.parameters()

        assert "blur_strength" in params
        assert "soft_threshold" in params

    @e2e_pytest_unit
    def test_get_outputs(self):
        self.blur.outputs = {"output": np.ones((2, 3, 4))}
        returned_value = self.blur._get_outputs()

        assert returned_value == "output"

    @e2e_pytest_unit
    def test_postprocess(self, mocker):
        self.blur.output_blob_name = "output"
        self.blur.soft_threshold = 0.5
        self.blur.blur_strength = 2
        fake_output = {"output": np.ones((2, 3, 4))}
        fake_metadata = {"original_shape": (2, 3, 4)}
        returned_value = self.blur.postprocess(outputs=fake_output, meta=fake_metadata)

        assert type(returned_value).__module__ == np.__name__
        assert fake_metadata["feature_vector"] is None
        assert fake_metadata["soft_prediction"] is not None
