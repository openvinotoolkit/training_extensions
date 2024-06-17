# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from model_api.adapters.openvino_adapter import OpenvinoAdapter
from model_api.models import ImageModel, SegmentationModel
from model_api.models.types import NumericalValue
from otx.algo.visual_prompting.openvino_models import VisualPromptingDecoder, VisualPromptingImageEncoder


class TestVisualPromptingImageEncoder:
    def test_parameters(self) -> None:
        """Test parameters."""
        params = VisualPromptingImageEncoder.parameters()

        assert params.get("resize_type").default_value == "fit_to_window"
        assert params.get("image_size").default_value == 1024

    def test_preproces(self, mocker) -> None:
        """Test preprocess."""
        mocker.patch.object(ImageModel, "__init__")
        image_encoder = VisualPromptingImageEncoder("adapter")
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


class TestVisualPromptingDecoder:
    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        mocker.patch.object(SegmentationModel, "__init__")
        mocker_model_adapter = mocker.Mock(spec=OpenvinoAdapter)
        self.decoder = VisualPromptingDecoder(mocker_model_adapter)
        self.decoder.image_size = 6

    def test_parameters(self) -> None:
        """Test parameters."""
        params = VisualPromptingDecoder.parameters()

        assert isinstance(params.get("image_size"), NumericalValue)
        assert params.get("image_size").default_value == 1024

    def test_get_outputs(self) -> None:
        """Test _get_outputs."""
        results = self.decoder._get_outputs()

        assert results == "upscaled_masks"

    @pytest.mark.parametrize(
        ("prompts", "prompt_type", "expected"),
        [
            (
                {
                    "bboxes": [np.array([[1, 1], [2, 2]])],
                    "points": [],
                    "labels": {"bboxes": [np.array(1)]},
                    "orig_size": (4, 4),
                },
                "bboxes",
                {
                    "point_coords": np.array([[[1.5, 1.5], [3.0, 3.0]]]),
                    "point_labels": np.array([[2.0, 3.0]]),
                },
            ),
            (
                {
                    "bboxes": [],
                    "points": [np.array([[1, 1]])],
                    "labels": {"points": [np.array(1)]},
                    "orig_size": (4, 4),
                },
                "points",
                {
                    "point_coords": np.array([[[1.5, 1.5]]]),
                    "point_labels": np.array([[1.0]]),
                },
            ),
        ],
    )
    def test_preprocess(self, prompts: dict[str, Any], prompt_type: str, expected: dict[str, Any]) -> None:
        """Test preprocess"""
        results = self.decoder.preprocess(prompts)

        assert isinstance(results, list)
        for i in range(len(results)):
            assert "point_coords" in results[i]
            assert np.all(results[i]["point_coords"] == expected["point_coords"])
            assert "point_labels" in results[i]
            assert np.all(results[i]["point_labels"] == expected["point_labels"])
            assert "mask_input" in results[i]
            assert np.all(results[i]["mask_input"] == self.decoder.mask_input)
            assert "has_mask_input" in results[i]
            assert np.all(results[i]["has_mask_input"] == self.decoder.has_mask_input)
            assert "orig_size" in results[i]
            assert np.all(results[i]["orig_size"] == prompts["orig_size"])
            assert "label" in results[i]
            assert np.all(results[i]["label"] == prompts["labels"][prompt_type][i])

    def test_apply_coords(self) -> None:
        """Test apply_coords."""
        coords = np.array([[[1, 1], [2, 2]]])
        original_size = (12, 12)

        results = self.decoder.apply_coords(coords, original_size)

        assert results.shape == (1, 2, 2)
        assert np.all(results == np.array([[[0.5, 0.5], [1.0, 1.0]]]))

    @pytest.mark.parametrize(
        ("old_h", "old_w", "image_size", "expected"),
        [
            (4, 3, 6, (6, 5)),
            (3, 4, 6, (5, 6)),
        ],
    )
    def test_get_preprocess_shape(self, old_h: int, old_w: int, image_size: int, expected: tuple[int]) -> None:
        """Test _get_preprocess_shape."""
        result = self.decoder._get_preprocess_shape(old_h, old_w, image_size)

        assert result == expected

    def test_get_inputs(self) -> None:
        """Test _get_inputs."""
        self.decoder.inputs = {"images": np.ones((1, 4, 4, 3))}

        returned_value = self.decoder._get_inputs()

        assert returned_value[0] == ["images"]

    def test_postprocess(self, mocker) -> None:
        """Test postprocess."""
        self.decoder.output_blob_name = "upscaled_masks"
        self.decoder.mask_threshold = 0.0
        self.decoder.blur_strength = 2
        fake_output = {"upscaled_masks": np.ones((1, 1, 4, 4)), "scores": 0.1}
        fake_metadata = {"orig_size": np.array([[6, 6]]), "label": [1]}

        returned_value = self.decoder.postprocess(outputs=fake_output, meta=fake_metadata)

        assert isinstance(returned_value, dict)
        assert "upscaled_masks" in returned_value
        assert returned_value["upscaled_masks"].shape == (1, 1, 4, 4)
        assert "scores" in returned_value
        assert returned_value["scores"] == 0.1
        assert "hard_prediction" in returned_value
        assert returned_value["hard_prediction"].shape == (1, 4, 4)
        assert "soft_prediction" in returned_value
        assert returned_value["soft_prediction"].shape == (1, 4, 4)
        assert np.all(returned_value["soft_prediction"] == 0.1 * np.ones((1, 4, 4)))
