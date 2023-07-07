"""Tests model wrappers for openvino task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
from openvino.model_api.models import SegmentationModel
from otx.api.entities.label import LabelEntity
import pytest
from openvino.model_api.adapters.openvino_adapter import OpenvinoAdapter
from openvino.model_api.models.types import NumericalValue
from otx.algorithms.segmentation.adapters.openvino.model_wrappers.blur import (
    BlurSegmentation,
)
from otx.algorithms.visual_prompting.adapters.openvino.model_wrappers import (
    Decoder,
    ImageEncoder,
)

from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestImageEncoder:
    @e2e_pytest_unit
    def test_parameters(self):
        """Test parameters."""
        params = ImageEncoder.parameters()

        assert params.get("resize_type").default_value == "fit_to_window"
        assert params.get("mean_values").default_value == [123.675, 116.28, 103.53]
        assert params.get("scale_values").default_value == [58.395, 57.12, 57.375]


class TestDecoder:
    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        class MockAdapter(OpenvinoAdapter):
            def __init__(self):
                pass

        mocker.patch.object(SegmentationModel, "__init__")
        model_adapter = MockAdapter()
        self.decoder = Decoder(model_adapter)
        self.decoder.image_size = 6

    @e2e_pytest_unit
    def test_preprocess(self):
        """Test preprocess"""
        bbox = np.array([[1, 1], [2, 2]])
        original_size = (4, 4)

        results = self.decoder.preprocess(bbox, original_size)

        assert isinstance(results, dict)
        assert "point_coords" in results
        assert results["point_coords"].shape == (1, 2, 2)
        assert "point_labels" in results
        assert results["point_labels"].shape == (1, 2)
        assert "mask_input" in results
        assert "has_mask_input" in results
        assert "orig_size" in results

    @e2e_pytest_unit
    def test_parameters(self):
        """Test parameters."""
        params = Decoder.parameters()

        assert isinstance(params.get("image_size"), NumericalValue)
        assert params.get("image_size").default_value == 1024

    @e2e_pytest_unit
    def test_get_inputs(self):
        """Test _get_inputs."""
        self.decoder.inputs = {"images": np.ones((2, 3, 4))}
        returned_value = self.decoder._get_inputs()

        assert returned_value[0] == ["images"]

    @e2e_pytest_unit
    def test_get_outputs(self):
        """Test _get_outputs."""
        self.decoder.outputs = {"low_res_masks": np.ones((2, 3, 4))}
        returned_value = self.decoder._get_outputs()

        assert returned_value == "low_res_masks"

    @e2e_pytest_unit
    def test_postprocess(self, mocker):
        """Test postprocess."""
        self.decoder.output_blob_name = "masks"
        self.decoder.soft_threshold = 0.5
        self.decoder.blur_strength = 2
        fake_output = {"masks": np.ones((4, 4)), "iou_predictions": 0.1}
        fake_metadata = {"original_size": np.array((6, 6)), "label": mocker.Mock(spec=LabelEntity)}
        returned_value = self.decoder.postprocess(outputs=fake_output, meta=fake_metadata)

        assert isinstance(returned_value, tuple)
        assert np.all(returned_value[0].shape == fake_metadata["original_size"])
        assert np.all(returned_value[1].shape == fake_metadata["original_size"])

    @e2e_pytest_unit
    def test_resize_and_crop(self, mocker):
        """Test resize_and_crop."""
        mocker.patch.object(self.decoder, "resize_longest_image_size", return_value=np.array((6, 6)))

        masks = np.zeros((2, 2))
        orig_size = np.array((8, 8))

        results = self.decoder.resize_and_crop(masks, orig_size)

        assert results.shape == tuple(orig_size)

    @e2e_pytest_unit
    def test_resize_longest_image_size(self):
        """Test resize_longest_image_size."""
        original_size = np.array((2, 4))
        longest_side = 6

        results = self.decoder.resize_longest_image_size(original_size, longest_side)

        assert np.all(results == np.array((3, 6)))
