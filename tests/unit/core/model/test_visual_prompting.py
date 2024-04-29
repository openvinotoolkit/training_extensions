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
from otx.core.data.entity.base import Points
from otx.core.data.entity.visual_prompting import (
    VisualPromptingBatchPredEntity,
    ZeroShotVisualPromptingBatchDataEntity,
    ZeroShotVisualPromptingBatchPredEntity,
)
from otx.core.exporter.visual_prompting import OTXVisualPromptingModelExporter
from otx.core.model.visual_prompting import (
    OTXVisualPromptingModel,
    OTXZeroShotVisualPromptingModel,
    OVVisualPromptingModel,
    OVZeroShotVisualPromptingModel,
    _inference_step,
    _inference_step_for_zero_shot,
)
from otx.core.types.export import TaskLevelExportParameters
from torchvision import tv_tensors


@pytest.fixture()
def otx_visual_prompting_model(mocker) -> OTXVisualPromptingModel:
    mocker.patch.object(OTXVisualPromptingModel, "_create_model")
    model = OTXVisualPromptingModel(label_info=1)
    model.model.image_size = 1024
    return model


@pytest.fixture()
def otx_zero_shot_visual_prompting_model(mocker) -> OTXZeroShotVisualPromptingModel:
    mocker.patch.object(OTXZeroShotVisualPromptingModel, "_create_model")
    model = OTXZeroShotVisualPromptingModel(label_info=1)
    model.model.image_size = 1024
    return model


def test_inference_step(mocker, otx_visual_prompting_model, fxt_vpm_data_entity) -> None:
    """Test _inference_step."""
    otx_visual_prompting_model.configure_metric()
    mocker.patch.object(otx_visual_prompting_model, "forward", return_value=fxt_vpm_data_entity[2])
    mocker_updates = {}
    for k, v in otx_visual_prompting_model.metric.items():
        mocker_updates[k] = mocker.patch.object(v, "update")

    _inference_step(otx_visual_prompting_model, otx_visual_prompting_model.metric, fxt_vpm_data_entity[1])

    for v in mocker_updates.values():
        v.assert_called_once()


def test_inference_step_for_zero_shot(mocker, otx_visual_prompting_model, fxt_zero_shot_vpm_data_entity) -> None:
    """Test _inference_step_for_zero_shot."""
    otx_visual_prompting_model.configure_metric()
    mocker.patch.object(otx_visual_prompting_model, "forward", return_value=fxt_zero_shot_vpm_data_entity[2])
    mocker_updates = {}
    for k, v in otx_visual_prompting_model.metric.items():
        mocker_updates[k] = mocker.patch.object(v, "update")

    _inference_step_for_zero_shot(
        otx_visual_prompting_model,
        otx_visual_prompting_model.metric,
        fxt_zero_shot_vpm_data_entity[1],
    )

    for v in mocker_updates.values():
        v.assert_called_once()


def test_inference_step_for_zero_shot_with_more_preds(
    mocker,
    otx_visual_prompting_model,
    fxt_zero_shot_vpm_data_entity,
) -> None:
    """Test _inference_step_for_zero_shot with more preds."""
    otx_visual_prompting_model.configure_metric()
    preds = {}
    for k, v in fxt_zero_shot_vpm_data_entity[2].__dict__.items():
        if k in ["batch_size", "polygons"]:
            preds[k] = v
        else:
            preds[k] = v * 2
    mocker.patch.object(
        otx_visual_prompting_model,
        "forward",
        return_value=ZeroShotVisualPromptingBatchPredEntity(**preds),
    )
    mocker_updates = {}
    for k, v in otx_visual_prompting_model.metric.items():
        mocker_updates[k] = mocker.patch.object(v, "update")

    _inference_step_for_zero_shot(
        otx_visual_prompting_model,
        otx_visual_prompting_model.metric,
        fxt_zero_shot_vpm_data_entity[1],
    )

    for v in mocker_updates.values():
        v.assert_called_once()


def test_inference_step_for_zero_shot_with_more_target(
    mocker,
    otx_visual_prompting_model,
    fxt_zero_shot_vpm_data_entity,
) -> None:
    """Test _inference_step_for_zero_shot with more target."""
    otx_visual_prompting_model.configure_metric()
    mocker.patch.object(otx_visual_prompting_model, "forward", return_value=fxt_zero_shot_vpm_data_entity[2])
    mocker_updates = {}
    for k, v in otx_visual_prompting_model.metric.items():
        mocker_updates[k] = mocker.patch.object(v, "update")
    target = {}
    for k, v in fxt_zero_shot_vpm_data_entity[1].__dict__.items():
        if k in ["batch_size"]:
            target[k] = v
        else:
            target[k] = v * 2

    _inference_step_for_zero_shot(
        otx_visual_prompting_model,
        otx_visual_prompting_model.metric,
        ZeroShotVisualPromptingBatchDataEntity(**target),
    )

    for v in mocker_updates.values():
        v.assert_called_once()


class TestOTXVisualPromptingModel:
    def test_exporter(self, otx_visual_prompting_model) -> None:
        """Test _exporter."""
        exporter = otx_visual_prompting_model._exporter
        assert isinstance(exporter, OTXVisualPromptingModelExporter)
        assert exporter.input_size == (1, 3, 1024, 1024)
        assert exporter.resize_mode == "fit_to_window"
        assert exporter.mean == (123.675, 116.28, 103.53)
        assert exporter.std == (58.395, 57.12, 57.375)

    def test_export_parameters(self, otx_visual_prompting_model) -> None:
        """Test _export_parameters."""
        export_parameters = otx_visual_prompting_model._export_parameters

        assert isinstance(export_parameters, TaskLevelExportParameters)
        assert export_parameters.model_type == "Visual_Prompting"
        assert export_parameters.task_type == "visual_prompting"

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


class TestOTXZeroShotVisualPromptingModel:
    def test_exporter(self, otx_zero_shot_visual_prompting_model) -> None:
        """Test _exporter."""
        exporter = otx_zero_shot_visual_prompting_model._exporter
        assert isinstance(exporter, OTXVisualPromptingModelExporter)
        assert exporter.input_size == (1, 3, 1024, 1024)
        assert exporter.resize_mode == "fit_to_window"
        assert exporter.mean == (123.675, 116.28, 103.53)
        assert exporter.std == (58.395, 57.12, 57.375)

    def test_export_parameters(self, otx_zero_shot_visual_prompting_model) -> None:
        """Test _export_parameters."""
        export_parameters = otx_zero_shot_visual_prompting_model._export_parameters

        assert isinstance(export_parameters, TaskLevelExportParameters)
        assert export_parameters.model_type == "Visual_Prompting"
        assert export_parameters.task_type == "visual_prompting"

    def test_optimization_config(self, otx_zero_shot_visual_prompting_model) -> None:
        """Test _optimization_config."""
        optimization_config = otx_zero_shot_visual_prompting_model._optimization_config

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

    def test_on_test_start(self, mocker, otx_zero_shot_visual_prompting_model) -> None:
        """Test on_test_start."""
        otx_zero_shot_visual_prompting_model.load_reference_info = Mock(return_value=False)
        otx_zero_shot_visual_prompting_model.trainer = Mock()
        mocker_run = mocker.patch.object(otx_zero_shot_visual_prompting_model.trainer.fit_loop, "run")
        mocker_setup_data = mocker.patch.object(
            otx_zero_shot_visual_prompting_model.trainer._evaluation_loop,
            "setup_data",
        )
        mocker_reset = mocker.patch.object(otx_zero_shot_visual_prompting_model.trainer._evaluation_loop, "reset")

        otx_zero_shot_visual_prompting_model.on_test_start()

        mocker_run.assert_called_once()
        mocker_setup_data.assert_called_once()
        mocker_reset.assert_called_once()

    def test_on_predict_start(self, mocker, otx_zero_shot_visual_prompting_model) -> None:
        """Test on_predict_start."""
        otx_zero_shot_visual_prompting_model.load_reference_info = Mock(return_value=False)
        otx_zero_shot_visual_prompting_model.trainer = Mock()
        mocker_run = mocker.patch.object(otx_zero_shot_visual_prompting_model.trainer.fit_loop, "run")
        mocker_setup_data = mocker.patch.object(
            otx_zero_shot_visual_prompting_model.trainer._evaluation_loop,
            "setup_data",
        )
        mocker_reset = mocker.patch.object(otx_zero_shot_visual_prompting_model.trainer._evaluation_loop, "reset")

        otx_zero_shot_visual_prompting_model.on_predict_start()

        mocker_run.assert_called_once()
        mocker_setup_data.assert_called_once()
        mocker_reset.assert_called_once()

    def test_on_train_epoch_end(self, mocker, tmpdir, otx_zero_shot_visual_prompting_model) -> None:
        """Test on_train_epoch_end."""
        otx_zero_shot_visual_prompting_model.save_outputs = True
        otx_zero_shot_visual_prompting_model.save_reference_info = Mock()
        otx_zero_shot_visual_prompting_model.trainer = Mock()
        mocker.patch.object(otx_zero_shot_visual_prompting_model.trainer, "default_root_dir")

        otx_zero_shot_visual_prompting_model.on_train_epoch_end()


class TestOVVisualPromptingModel:
    @pytest.fixture()
    def set_ov_visual_prompting_model(self, mocker, tmpdir):
        def ov_visual_prompting_model(for_create_model: bool = False) -> OVVisualPromptingModel:
            if for_create_model:
                mocker.patch("model_api.adapters.create_core")
                mocker.patch("model_api.adapters.get_user_config")
                mocker.patch("model_api.adapters.OpenvinoAdapter")
                mocker.patch("model_api.models.Model.create_model")
            else:
                mocker.patch.object(
                    OVVisualPromptingModel,
                    "_create_model",
                    return_value={"image_encoder": Mock(), "decoder": Mock()},
                )
            dirpath = Path(tmpdir)
            (dirpath / "exported_model_image_encoder.xml").touch()
            (dirpath / "exported_model_decoder.xml").touch()
            model_name = str(dirpath / "exported_model_decoder.xml")
            return OVVisualPromptingModel(num_classes=0, model_name=model_name)

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
    def ov_zero_shot_visual_prompting_model(self, mocker, tmpdir) -> OVZeroShotVisualPromptingModel:
        mocker.patch.object(
            OVZeroShotVisualPromptingModel,
            "_create_model",
            return_value={"image_encoder": Mock(), "decoder": Mock()},
        )
        mocker.patch.object(OVZeroShotVisualPromptingModel, "initialize_reference_info")
        dirpath = Path(tmpdir)
        (dirpath / "exported_model_image_encoder.xml").touch()
        (dirpath / "exported_model_decoder.xml").touch()
        model_name = str(dirpath / "exported_model_decoder.xml")

        return OVZeroShotVisualPromptingModel(num_classes=0, model_name=model_name)

    @pytest.mark.parametrize("training", [True, False])
    def test_forward(
        self,
        mocker,
        ov_zero_shot_visual_prompting_model,
        fxt_zero_shot_vpm_data_entity,
        training: bool,
    ) -> None:
        """Test forward."""
        ov_zero_shot_visual_prompting_model.training = training
        ov_zero_shot_visual_prompting_model.reference_feats = "reference_feats"
        ov_zero_shot_visual_prompting_model.used_indices = "used_indices"
        mocker_fn = mocker.patch.object(ov_zero_shot_visual_prompting_model, "learn" if training else "infer")
        mocker_customize_outputs = mocker.patch.object(ov_zero_shot_visual_prompting_model, "_customize_outputs")

        ov_zero_shot_visual_prompting_model.forward(fxt_zero_shot_vpm_data_entity[1])

        mocker_fn.assert_called_once()
        mocker_customize_outputs.assert_called_once()

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
                "upscaled_masks": np.random.randn(1, 4, 1024, 1024),
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
            return_value=np.random.rand(1, 256),
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
                "upscaled_masks": np.random.randn(1, 4, 1024, 1024),
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

    def test_customize_outputs_training(
        self,
        ov_zero_shot_visual_prompting_model,
        fxt_zero_shot_vpm_data_entity,
    ) -> None:
        ov_zero_shot_visual_prompting_model.training = True

        outputs = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]

        result = ov_zero_shot_visual_prompting_model._customize_outputs(outputs, fxt_zero_shot_vpm_data_entity[1])

        assert result == outputs

    def test_customize_outputs_inference(
        self,
        ov_zero_shot_visual_prompting_model,
        fxt_zero_shot_vpm_data_entity,
    ) -> None:
        ov_zero_shot_visual_prompting_model.training = False

        outputs = [
            ({1: [[1, 2, 3], [4, 5, 6]]}, {1: [[7, 8, 9], [10, 11, 12]]}),
            ({2: [[13, 14, 15], [16, 17, 18]]}, {2: [[19, 20, 21], [22, 23, 24]]}),
        ]

        result = ov_zero_shot_visual_prompting_model._customize_outputs(outputs, fxt_zero_shot_vpm_data_entity[1])

        assert isinstance(result, ZeroShotVisualPromptingBatchPredEntity)
        assert result.batch_size == len(outputs)
        assert result.images == fxt_zero_shot_vpm_data_entity[1].images
        assert result.imgs_info == fxt_zero_shot_vpm_data_entity[1].imgs_info

        assert isinstance(result.masks, list)
        assert all(isinstance(mask, tv_tensors.Mask) for mask in result.masks)

        assert isinstance(result.prompts, list)
        assert all(isinstance(prompt, Points) for prompt in result.prompts)

        assert isinstance(result.scores, list)
        assert all(isinstance(score, torch.Tensor) for score in result.scores)

        assert isinstance(result.labels, list)
        assert all(isinstance(label, torch.LongTensor) for label in result.labels)

        assert result.polygons == []

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

    def test_load_reference_info(self, mocker, ov_zero_shot_visual_prompting_model) -> None:
        """Test load_latest_reference_info."""
        ov_zero_shot_visual_prompting_model.model["decoder"].embed_dim = 256

        # get previously saved reference info
        mocker.patch(
            "pickle.load",
            return_value={"reference_feats": np.zeros((1, 1, 256)), "used_indices": np.array([0])},
        )
        mocker.patch("pathlib.Path.is_file", return_value=True)
        mocker.patch("pathlib.Path.open", return_value="Mocked data")

        ov_zero_shot_visual_prompting_model.load_reference_info(".")
        assert ov_zero_shot_visual_prompting_model.reference_feats.shape == (1, 1, 256)
        assert ov_zero_shot_visual_prompting_model.used_indices.shape == (1,)

        # no saved reference info
        mocker.patch("pathlib.Path.is_file", return_value=False)

        ov_zero_shot_visual_prompting_model.reference_feats = np.zeros((0, 1, 256), dtype=np.float32)
        ov_zero_shot_visual_prompting_model.used_indices = np.array([], dtype=np.int64)
        ov_zero_shot_visual_prompting_model.load_reference_info(".")

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
