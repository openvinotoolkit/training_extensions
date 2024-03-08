"""Tests model wrappers for openvino task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Tuple, Dict, Any

import numpy as np
import pytest
from openvino.model_api.adapters.openvino_adapter import OpenvinoAdapter
from openvino.model_api.models import ImageModel, SegmentationModel
from openvino.model_api.models.types import NumericalValue

from otx.algorithms.visual_prompting.adapters.openvino.model_wrappers import (
    Decoder,
    ImageEncoder,
)
from otx.api.entities.label import LabelEntity
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestImageEncoder:
    @e2e_pytest_unit
    def test_parameters(self):
        """Test parameters."""
        params = ImageEncoder.parameters()

        assert params.get("resize_type").default_value == "fit_to_window"

    @e2e_pytest_unit
    def test_preproces(self, mocker):
        """Test preprocess."""
        mocker.patch.object(ImageModel, "__init__")
        image_encoder = ImageEncoder("adapter")
        fake_inputs = np.ones((4, 4, 3))
        image_encoder.h, image_encoder.w, image_encoder.c = fake_inputs.shape
        image_encoder.image_blob_name = "images"
        image_encoder.resize_type = "fit_to_window"

        dict_inputs, meta = image_encoder.preprocess(fake_inputs)

        assert dict_inputs["images"].shape == (1, 4, 4, 3)
        assert meta["original_shape"] == (4, 4, 3)
        assert meta["resized_shape"] == (4, 4, 3)
        assert "resize_type" in meta
        assert meta["resize_type"] == "fit_to_window"


class TestDecoder:
    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        mocker.patch.object(SegmentationModel, "__init__")
        mocker_model_adapter = mocker.Mock(spec=OpenvinoAdapter)
        self.decoder = Decoder(mocker_model_adapter)
        self.decoder.image_size = 6

    @e2e_pytest_unit
    def test_parameters(self):
        """Test parameters."""
        params = Decoder.parameters()

        assert isinstance(params.get("image_size"), NumericalValue)
        assert params.get("image_size").default_value == 1024

    @e2e_pytest_unit
    def test_get_outputs(self):
        """Test _get_outputs."""
        results = self.decoder._get_outputs()

        assert "upscaled_masks" == results

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "prompts,expected",
        [
            (
                {
                    "bboxes": [np.array([[1, 1], [2, 2]])],
                    "points": [],
                    "labels": {"bboxes": [1]},
                    "original_size": (4, 4),
                },
                {
                    "point_coords": (1, 2, 2),
                    "point_labels": (1, 2),
                },
            ),
            (
                {"bboxes": [], "points": [np.array([[1, 1]])], "labels": {"points": [1]}, "original_size": (4, 4)},
                {
                    "point_coords": (1, 1, 2),
                    "point_labels": (1, 1),
                },
            ),
        ],
    )
    def test_preprocess(self, prompts: Dict[str, Any], expected: Dict[str, Any]):
        """Test preprocess"""
        results = self.decoder.preprocess(prompts, {})

        assert isinstance(results, list)
        assert "point_coords" in results[0]
        assert results[0]["point_coords"].shape == expected["point_coords"]
        assert "point_labels" in results[0]
        assert results[0]["point_labels"].shape == expected["point_labels"]
        assert "mask_input" in results[0]
        assert "has_mask_input" in results[0]
        assert "orig_size" in results[0]

    @e2e_pytest_unit
    def test_apply_coords(self):
        """Test _apply_coords."""
        coords = np.array([[[1, 1], [2, 2]]])
        original_size = (12, 12)

        results = self.decoder._apply_coords(coords, original_size)

        assert results.shape == (1, 2, 2)
        assert np.all(results == np.array([[[0.5, 0.5], [1.0, 1.0]]]))

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "old_h,old_w,image_size,expected",
        [
            (4, 3, 6, (6, 5)),
            (3, 4, 6, (5, 6)),
        ],
    )
    def test_get_preprocess_shape(self, old_h: int, old_w: int, image_size: int, expected: Tuple[int]):
        """Test _get_preprocess_shape."""
        result = self.decoder._get_preprocess_shape(old_h, old_w, image_size)

        assert result == expected

    @e2e_pytest_unit
    def test_get_inputs(self):
        """Test _get_inputs."""
        self.decoder.inputs = {"images": np.ones((1, 4, 4, 3))}

        returned_value = self.decoder._get_inputs()

        assert returned_value[0] == ["images"]

    @e2e_pytest_unit
    def test_postprocess(self, mocker):
        """Test postprocess."""
        self.decoder.output_blob_name = "upscaled_masks"
        self.decoder.mask_threshold = 0.0
        self.decoder.blur_strength = 2
        fake_output = {"upscaled_masks": np.ones((4, 4)), "scores": 0.1}
        fake_metadata = {"original_size": np.array([[6, 6]]), "label": mocker.Mock(spec=LabelEntity)}
        returned_value = self.decoder.postprocess(outputs=fake_output, meta=fake_metadata)

        assert isinstance(returned_value, tuple)
