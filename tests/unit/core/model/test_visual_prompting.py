# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for visual prompting model entity."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from otx.core.data.entity.visual_prompting import VisualPromptingBatchPredEntity
from otx.core.exporter.visual_prompting import OTXVisualPromptingModelExporter
from otx.core.model.visual_prompting import (
    OTXVisualPromptingModel,
    OVVisualPromptingModel,
    OVZeroShotVisualPromptingModel,
)
from torchvision import tv_tensors


class TestOTXVisualPromptingModel:
    @pytest.fixture()
    def otx_visual_prompting_model(self, mocker) -> OTXVisualPromptingModel:
        mocker.patch.object(OTXVisualPromptingModel, "_create_model")
        return OTXVisualPromptingModel(num_classes=1)

    def test_exporter(self, otx_visual_prompting_model) -> None:
        """Test _exporter."""
        assert isinstance(otx_visual_prompting_model._exporter, OTXVisualPromptingModelExporter)

    def test_export_parameters(self, otx_visual_prompting_model) -> None:
        """Test _export_parameters."""
        otx_visual_prompting_model.model.image_size = 1024

        export_parameters = otx_visual_prompting_model._export_parameters

        assert export_parameters["input_size"] == (1, 3, 1024, 1024)
        assert export_parameters["resize_mode"] == "fit_to_window"
        assert export_parameters["mean"] == (123.675, 116.28, 103.53)
        assert export_parameters["std"] == (58.395, 57.12, 57.375)

    def test_optimization_config(self, otx_visual_prompting_model) -> None:
        """Test _optimization_config."""
        optimization_config = otx_visual_prompting_model._optimization_config

        assert optimization_config == {
            "model_type": "transformer",
            "advanced_parameters": {
                "activations_range_estimator_params": {
                    "min": {
                        "statistics_type": "QUANTILE",
                        "aggregator_type": "MIN",
                        "quantile_outlier_prob": "1e-4",
                    },
                    "max": {
                        "statistics_type": "QUANTILE",
                        "aggregator_type": "MAX",
                        "quantile_outlier_prob": "1e-4",
                    },
                },
            },
        }


class TestOVVisualPromptingModel:
    @pytest.fixture()
    def set_ov_visual_prompting_model(self, mocker):
        def ov_visual_prompting_model(for_create_model: bool = False) -> OVVisualPromptingModel:
            if for_create_model:
                mocker.patch("openvino.model_api.adapters.create_core")
                mocker.patch("openvino.model_api.adapters.get_user_config")
                mocker.patch("openvino.model_api.adapters.OpenvinoAdapter")
                mocker.patch("openvino.model_api.models.Model.create_model")
            else:
                mocker.patch.object(
                    OVVisualPromptingModel,
                    "_create_model",
                    return_value={"image_encoder": Mock(), "decoder": Mock()},
                )
            return OVVisualPromptingModel(num_classes=0, model_name="exported_model_decoder.xml")

        return ov_visual_prompting_model

    def test_create_model(self, set_ov_visual_prompting_model) -> None:
        """Test _create_model."""
        ov_visual_prompting_model = set_ov_visual_prompting_model(for_create_model=True)
        ov_models = ov_visual_prompting_model._create_model()

        assert isinstance(ov_models, dict)
        assert "image_encoder" in ov_models
        assert "decoder" in ov_models

    def test_forward(self, mocker, set_ov_visual_prompting_model, fxt_vpm_data_entity) -> None:
        """Test forward."""
        ov_visual_prompting_model = set_ov_visual_prompting_model()
        mocker.patch.object(
            ov_visual_prompting_model.model["image_encoder"],
            "preprocess",
            return_value=(np.zeros((1, 3, 1024, 1024)), {}),
        )
        mocker.patch.object(
            ov_visual_prompting_model.model["image_encoder"],
            "infer_sync",
            return_value={"image_embeddings": np.random.random((1, 256, 64, 64))},
        )
        mocker.patch.object(
            ov_visual_prompting_model.model["decoder"],
            "preprocess",
            return_value=[
                {
                    "point_coords": np.array([1, 1]).reshape(-1, 1, 2),
                    "point_labels": np.array([1], dtype=np.float32).reshape(-1, 1),
                    "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
                    "has_mask_input": np.zeros((1, 1), dtype=np.float32),
                    "orig_size": np.array([1024, 1024], dtype=np.int64).reshape(-1, 2),
                    "label": 1,
                },
            ],
        )
        mocker.patch.object(
            ov_visual_prompting_model.model["decoder"],
            "infer_sync",
            return_value={
                "iou_predictions": 0.0,
                "upscaled_masks": np.zeros((1, 1, 1024, 1024), dtype=np.float32),
            },
        )
        mocker.patch.object(
            ov_visual_prompting_model.model["decoder"],
            "postprocess",
            return_value={
                "hard_prediction": np.zeros((1, 1, 1024, 1024), dtype=np.float32),
                "soft_prediction": np.zeros((1, 1, 1024, 1024), dtype=np.float32),
                "scores": np.zeros((1, 1), dtype=np.float32),
            },
        )

        results = ov_visual_prompting_model(fxt_vpm_data_entity[1])

        assert isinstance(results, VisualPromptingBatchPredEntity)
        assert isinstance(results.images, list)
        assert isinstance(results.images[0], tv_tensors.Image)
        assert isinstance(results.masks, list)
        assert isinstance(results.masks[0], tv_tensors.Mask)

    def test_optimize(self, tmpdir, mocker, set_ov_visual_prompting_model) -> None:
        """Test optimize."""
        mocker.patch("openvino.Core.read_model")
        mocker.patch("openvino.save_model")
        mocker.patch("nncf.quantize")

        ov_visual_prompting_model = set_ov_visual_prompting_model()
        fake_data_module = Mock()

        results = ov_visual_prompting_model.optimize(tmpdir, fake_data_module)

        assert "image_encoder" in results
        assert "decoder" in results


class TestOVZeroShotVisualPromptingModel:
    @pytest.fixture()
    def ov_zero_shot_visual_prompting_model(self, mocker) -> OVZeroShotVisualPromptingModel:
        mocker.patch.object(
            OVZeroShotVisualPromptingModel,
            "_create_model",
            return_value={"image_encoder": Mock(), "decoder": Mock()},
        )
        mocker.patch.object(OVZeroShotVisualPromptingModel, "initialize_reference_info")
        return OVZeroShotVisualPromptingModel(num_classes=0, model_name="exported_model_decoder.xml")

    def test_learn(self, mocker, ov_zero_shot_visual_prompting_model, fxt_zero_shot_vpm_data_entity) -> None:
        """Test learn."""
        ov_zero_shot_visual_prompting_model.reference_feats = np.zeros((0, 1, 256), dtype=np.float32)
        ov_zero_shot_visual_prompting_model.used_indices = np.array([], dtype=np.int64)
        ov_zero_shot_visual_prompting_model.model["decoder"].mask_threshold = 0.0
        mocker.patch.object(
            ov_zero_shot_visual_prompting_model.model["image_encoder"],
            "preprocess",
            return_value=(np.zeros((1, 3, 1024, 1024)), {"original_shape": (1024, 1024)}),
        )
        mocker.patch.object(
            ov_zero_shot_visual_prompting_model.model["image_encoder"],
            "infer_sync",
            return_value={"image_embeddings": np.random.random((1, 256, 64, 64))},
        )
        mocker.patch.object(
            ov_zero_shot_visual_prompting_model.model["decoder"],
            "preprocess",
            return_value=[
                {
                    "point_coords": np.array([1, 1]).reshape(-1, 1, 2),
                    "point_labels": np.array([1], dtype=np.float32).reshape(-1, 1),
                    "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
                    "has_mask_input": np.zeros((1, 1), dtype=np.float32),
                    "orig_size": np.array([1024, 1024], dtype=np.int64).reshape(-1, 2),
                    "label": np.array([1], dtype=np.int64),
                },
            ],
        )
        mocker.patch.object(
            ov_zero_shot_visual_prompting_model.model["decoder"],
            "infer_sync",
            return_value={
                "iou_predictions": np.array([[0.1, 0.3, 0.5, 0.7]]),
                "upscaled_masks": np.random.randn(1, 4, 1024, 1024),  # noqa: NPY002
                "low_res_masks": np.zeros((1, 4, 64, 64), dtype=np.float32),
            },
        )
        mocker.patch.object(
            ov_zero_shot_visual_prompting_model.model["decoder"],
            "postprocess",
            return_value={
                "hard_prediction": np.zeros((1, 1, 1024, 1024), dtype=np.float32),
                "soft_prediction": np.zeros((1, 1, 1024, 1024), dtype=np.float32),
                "scores": np.zeros((1, 1), dtype=np.float32),
            },
        )
        mocker.patch.object(
            ov_zero_shot_visual_prompting_model,
            "_generate_masked_features",
            return_value=np.random.rand(1, 256),  # noqa: NPY002
        )
        reference_info, ref_masks = ov_zero_shot_visual_prompting_model.learn(
            inputs=fxt_zero_shot_vpm_data_entity[1],
            reset_feat=True,
        )

        assert reference_info["reference_feats"].shape == torch.Size((2, 1, 256))
        assert 1 in reference_info["used_indices"]
        assert ref_masks[0].shape == torch.Size((2, 1024, 1024))

    def test_infer(self, mocker, ov_zero_shot_visual_prompting_model, fxt_zero_shot_vpm_data_entity) -> None:
        """Test infer."""
        ov_zero_shot_visual_prompting_model.model["decoder"].mask_threshold = 0.0
        ov_zero_shot_visual_prompting_model.model["decoder"].output_blob_name = "upscaled_masks"
        mocker.patch.object(
            ov_zero_shot_visual_prompting_model.model["image_encoder"],
            "preprocess",
            return_value=(np.zeros((1, 3, 1024, 1024)), {"original_shape": (1024, 1024)}),
        )
        mocker.patch.object(
            ov_zero_shot_visual_prompting_model.model["image_encoder"],
            "infer_sync",
            return_value={"image_embeddings": np.random.random((1, 256, 64, 64))},
        )
        mocker.patch.object(
            ov_zero_shot_visual_prompting_model.model["decoder"],
            "preprocess",
            return_value=[
                {
                    "point_coords": np.array([1, 1]).reshape(-1, 1, 2),
                    "point_labels": np.array([1], dtype=np.float32).reshape(-1, 1),
                    "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
                    "has_mask_input": np.zeros((1, 1), dtype=np.float32),
                    "orig_size": np.array([1024, 1024], dtype=np.int64).reshape(-1, 2),
                    "label": np.array([1], dtype=np.int64),
                },
            ],
        )
        mocker.patch.object(
            ov_zero_shot_visual_prompting_model.model["decoder"],
            "infer_sync",
            return_value={
                "iou_predictions": np.array([[0.1, 0.3, 0.5, 0.7]]),
                "upscaled_masks": np.random.randn(1, 4, 1024, 1024),  # noqa: NPY002
                "low_res_masks": np.zeros((1, 4, 64, 64), dtype=np.float32),
            },
        )
        mocker.patch.object(
            ov_zero_shot_visual_prompting_model.model["decoder"],
            "postprocess",
            return_value={
                "hard_prediction": np.zeros((1, 1, 1024, 1024), dtype=np.float32),
                "soft_prediction": np.zeros((1, 1, 1024, 1024), dtype=np.float32),
                "scores": np.zeros((1, 1), dtype=np.float32),
            },
        )
        mocker.patch.object(
            ov_zero_shot_visual_prompting_model.model["decoder"],
            "apply_coords",
            return_value=np.array([[1, 1], [2, 2]]),
        )
        mocker.patch.object(
            ov_zero_shot_visual_prompting_model,
            "_get_prompt_candidates",
            return_value=({1: np.array([[1, 1, 0.5]])}, {1: np.array([[2, 2]])}),
        )

        reference_feats = torch.rand(2, 1, 256)
        used_indices = np.array([1])

        results = ov_zero_shot_visual_prompting_model.infer(
            inputs=fxt_zero_shot_vpm_data_entity[1],
            reference_feats=reference_feats,
            used_indices=used_indices,
        )

        for predicted_masks, used_points in results:
            for label, predicted_mask in predicted_masks.items():
                for pm, _ in zip(predicted_mask, used_points[label]):
                    assert pm.shape == (1024, 1024)

    def test_gather_prompts_with_labels(self, ov_zero_shot_visual_prompting_model) -> None:
        """Test _gather_prompts_with_labels."""
        batch_prompts = [
            [
                {"bboxes": "bboxes", "label": 1},
                {"points": "points", "label": 2},
            ],
        ]

        processed_prompts = ov_zero_shot_visual_prompting_model._gather_prompts_with_labels(batch_prompts)

        for prompts in processed_prompts:
            for label, prompt in prompts.items():
                if label == 1:
                    assert "bboxes" in prompt[0]
                else:
                    assert "points" in prompt[0]
                assert prompt[0]["label"] == label

    def test_initialize_reference_info(self, ov_zero_shot_visual_prompting_model) -> None:
        """Test initialize_reference_info."""
        ov_zero_shot_visual_prompting_model.reference_feats = np.zeros((0, 1, 256), dtype=np.float32)
        ov_zero_shot_visual_prompting_model.used_indices = np.array([], dtype=np.int64)

        assert ov_zero_shot_visual_prompting_model.reference_feats.shape == (0, 1, 256)
        assert ov_zero_shot_visual_prompting_model.used_indices.shape == (0,)

    @pytest.mark.parametrize("new_largest_label", [0, 3])
    def test_expand_reference_info(self, ov_zero_shot_visual_prompting_model, new_largest_label: int) -> None:
        """Test expand_reference_info."""
        ov_zero_shot_visual_prompting_model.reference_feats = np.zeros((0, 1, 256))

        ov_zero_shot_visual_prompting_model.expand_reference_info(
            new_largest_label=new_largest_label,
        )

        assert len(ov_zero_shot_visual_prompting_model.reference_feats) == new_largest_label + 1

    def test_generate_masked_features(self, ov_zero_shot_visual_prompting_model) -> None:
        """Test _generate_masked_features."""
        feats = np.random.random((8, 8, 1))
        masks = np.zeros((16, 16), dtype=np.float32)
        masks[4:12, 4:12] = 1.0

        masked_feat = ov_zero_shot_visual_prompting_model._generate_masked_features(
            feats=feats,
            masks=masks,
            threshold_mask=0.3,
            image_size=16,
        )

        assert masked_feat.shape == (1, 1)

    def test_pad_to_square(self, ov_zero_shot_visual_prompting_model) -> None:
        """Test _pad_to_square."""
        result = ov_zero_shot_visual_prompting_model._pad_to_square(x=np.ones((8, 8)), image_size=16)

        assert result[:8, :8].sum() == 8**2
        assert result[:8, 8:].sum() == 0
        assert result[8:, :8].sum() == 0
        assert result[8:, 8:].sum() == 0

    def test_find_latest_reference_info(self, mocker, ov_zero_shot_visual_prompting_model) -> None:
        """Test _find_latest_reference_info."""
        mocker.patch(
            "otx.core.model.visual_prompting.os.path.isdir",
            return_value=True,
        )

        # there are some saved reference info
        mocker.patch(
            "otx.core.model.visual_prompting.os.listdir",
            return_value=["1", "2"],
        )
        results = ov_zero_shot_visual_prompting_model._find_latest_reference_info(Path())
        assert results == "2"

        # there are no saved reference info
        mocker.patch(
            "otx.core.model.visual_prompting.os.listdir",
            return_value=[],
        )
        results = ov_zero_shot_visual_prompting_model._find_latest_reference_info(Path())
        assert results is None

    def test_load_latest_reference_info(self, mocker, ov_zero_shot_visual_prompting_model) -> None:
        """Test load_latest_reference_info."""
        ov_zero_shot_visual_prompting_model.model["decoder"].embed_dim = 256

        # get previously saved reference info
        mocker.patch.object(ov_zero_shot_visual_prompting_model, "_find_latest_reference_info", return_value="1")
        mocker.patch(
            "otx.core.model.visual_prompting.pickle.load",
            return_value={"reference_feats": np.zeros((1, 1, 256)), "used_indices": np.array([0])},
        )
        mocker.patch("otx.core.model.visual_prompting.Path.open", return_value="Mocked data")

        ov_zero_shot_visual_prompting_model.load_latest_reference_info()
        assert ov_zero_shot_visual_prompting_model.reference_feats.shape == (1, 1, 256)
        assert ov_zero_shot_visual_prompting_model.used_indices.shape == (1,)

        # no saved reference info
        mocker.patch.object(ov_zero_shot_visual_prompting_model, "_find_latest_reference_info", return_value=None)

        ov_zero_shot_visual_prompting_model.reference_feats = np.zeros((0, 1, 256), dtype=np.float32)
        ov_zero_shot_visual_prompting_model.used_indices = np.array([], dtype=np.int64)
        ov_zero_shot_visual_prompting_model.load_latest_reference_info()

        assert ov_zero_shot_visual_prompting_model.reference_feats.shape == (0, 1, 256)
        assert ov_zero_shot_visual_prompting_model.used_indices.shape == (0,)

    @pytest.mark.parametrize(
        "result_point_selection",
        [np.array([[2, 2, 0.9], [1, 2, 0.8], [0, 2, 0.7], [2, 1, 0.6]]), np.array([[-1, -1, -1]])],
    )
    def test_get_prompt_candidates(
        self,
        mocker,
        ov_zero_shot_visual_prompting_model,
        result_point_selection: np.ndarray,
    ) -> None:
        """Test get_prompt_candidates."""
        mocker.patch.object(
            ov_zero_shot_visual_prompting_model,
            "_point_selection",
            return_value=(result_point_selection, torch.zeros(1, 2)),
        )
        image_embeddings = np.ones((1, 4, 4, 4))
        reference_feats = np.random.random((1, 1, 4))
        used_indices = np.array([0])
        original_shape = np.array([3, 3], dtype=np.int64)

        total_points_scores, total_bg_coords = ov_zero_shot_visual_prompting_model._get_prompt_candidates(
            image_embeddings=image_embeddings,
            reference_feats=reference_feats,
            used_indices=used_indices,
            original_shape=original_shape,
        )

        assert total_points_scores[0].shape[0] == len(result_point_selection)
        assert total_bg_coords[0].shape[0] == 1

    @pytest.mark.parametrize(
        ("mask_sim", "expected"),
        [
            (
                np.arange(0.1, 1.0, 0.1).reshape(3, 3),
                np.array([[2, 2, 0.9], [1, 2, 0.8], [0, 2, 0.7], [2, 1, 0.6]]),
            ),
            (np.zeros((3, 3)), None),
        ],
    )
    def test_point_selection(
        self,
        ov_zero_shot_visual_prompting_model,
        mask_sim: np.ndarray,
        expected: np.ndarray,
    ) -> None:
        """Test _point_selection."""
        points_scores, bg_coords = ov_zero_shot_visual_prompting_model._point_selection(
            mask_sim=mask_sim,
            original_shape=np.array([3, 3]),
            threshold=np.array([[0.5]]),
            num_bg_points=1,
        )

        if points_scores is not None:
            assert np.allclose(points_scores, expected)

    def test_resize_to_original_shape(self, ov_zero_shot_visual_prompting_model) -> None:
        """Test _resize_to_original_shape."""
        masks = np.random.random((8, 8))
        image_size = 6
        original_shape = np.array([8, 10], dtype=np.int64)

        resized_masks = ov_zero_shot_visual_prompting_model._resize_to_original_shape(masks, image_size, original_shape)

        assert isinstance(resized_masks, np.ndarray)
        assert resized_masks.shape == (8, 10)

    def test_get_prepadded_size(self, ov_zero_shot_visual_prompting_model) -> None:
        """Test _get_prepadded_size."""
        original_shape = np.array([8, 10], dtype=np.int64)
        image_size = 6

        prepadded_size = ov_zero_shot_visual_prompting_model._get_prepadded_size(original_shape, image_size)

        assert isinstance(prepadded_size, np.ndarray)
        assert prepadded_size.dtype == np.int64
        assert prepadded_size.shape == (2,)
        assert np.all(prepadded_size == np.array([5, 6], dtype=np.int64))

    def test_inspect_overlapping_areas(self, mocker, ov_zero_shot_visual_prompting_model) -> None:
        """Test _inspect_overlapping_areas."""
        predicted_masks = {
            0: [
                np.array(
                    [
                        [1, 1, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ],
                ),
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0],
                        [0, 0, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ],
                ),
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 0, 0],
                        [0, 1, 1, 1, 0, 0],
                    ],
                ),
            ],
            1: [
                np.array(
                    [
                        [0, 0, 0, 1, 1, 0],
                        [0, 0, 0, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ],
                ),
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1],
                        [0, 0, 0, 0, 1, 1],
                    ],
                ),
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 0, 0, 0],
                        [0, 1, 1, 0, 0, 0],
                    ],
                ),
                np.array(
                    [
                        [1, 1, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ],
                ),
            ],
        }
        used_points = {
            0: [
                np.array([0, 0, 0.5]),  # to be removed
                np.array([2, 2, 0.5]),
                np.array([1, 4, 0.5]),
            ],
            1: [
                np.array([3, 0, 0.5]),
                np.array([4, 4, 0.5]),
                np.array([1, 4, 0.3]),  # to be removed
                np.array([0, 0, 0.7]),
            ],
        }

        ov_zero_shot_visual_prompting_model._inspect_overlapping_areas(predicted_masks, used_points, threshold_iou=0.5)

        assert len(predicted_masks[0]) == 2
        assert len(predicted_masks[1]) == 3
        assert all(np.array([2, 2, 0.5]) == used_points[0][0])
        assert all(np.array([0, 0, 0.7]) == used_points[1][2])

    @pytest.mark.parametrize(
        ("largest", "expected_scores", "expected_ind"),
        [
            (True, np.array([[3, 2], [6, 5], [9, 8]]), np.array([[2, 1], [2, 1], [2, 1]])),
            (False, np.array([[1, 2], [4, 5], [7, 8]]), np.array([[0, 1], [0, 1], [0, 1]])),
        ],
    )
    def test_topk_numpy(
        self,
        ov_zero_shot_visual_prompting_model,
        largest: bool,
        expected_scores: np.ndarray,
        expected_ind: np.ndarray,
    ) -> None:
        """Test _topk_numpy."""
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        k = 2
        axis = -1

        scores, ind = ov_zero_shot_visual_prompting_model._topk_numpy(x, k, axis, largest)

        np.testing.assert_array_equal(scores, expected_scores)
        np.testing.assert_array_equal(ind, expected_ind)
