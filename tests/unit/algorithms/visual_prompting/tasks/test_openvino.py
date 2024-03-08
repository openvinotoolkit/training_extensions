"""Tests the methods in the OpenVINO task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from copy import deepcopy
from typing import Optional, Dict, Tuple

import os
import numpy as np
import pytest
from otx.api.usecases.tasks.interfaces.optimization_interface import OptimizationType
import torch
from openvino.model_api.models import Model
from otx.api.entities.subset import Subset

from otx.algorithms.visual_prompting.adapters.pytorch_lightning.datasets.dataset import (
    OTXVisualPromptingDataset,
)
from otx.algorithms.visual_prompting.configs.base import VisualPromptingBaseConfig
from otx.algorithms.visual_prompting.tasks.openvino import (
    OpenVINOVisualPromptingInferencer,
    OpenVINOZeroShotVisualPromptingInferencer,
    OpenVINOVisualPromptingTask,
    OpenVINOZeroShotVisualPromptingTask,
    OTXOpenVinoDataLoader,
)
from otx.api.configuration.configurable_parameters import ConfigurableParameters
from otx.api.entities.annotation import Annotation
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.label import LabelEntity
from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.entities.metrics import Performance, ScoreMetric
from otx.api.entities.model import ModelConfiguration, ModelEntity
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.polygon import Point, Polygon
from otx.api.usecases.evaluation.metrics_helper import MetricsHelper
from otx.api.usecases.exportable_code.prediction_to_annotation_converter import (
    VisualPromptingToAnnotationConverter,
)
from otx.api.utils.shape_factory import ShapeFactory
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.visual_prompting.test_helpers import (
    generate_visual_prompting_dataset,
    init_environment,
)
from tests.unit.algorithms.visual_prompting.test_helpers import MockScoredLabel


class TestOpenVINOVisualPromptingInferencer:
    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        self.fake_annotation = [
            Annotation(
                Polygon(points=[Point(0, 0)]),
                id=0,
                labels=[ScoredLabel(LabelEntity(name="fake", domain="VISUALPROMPTING"), probability=1.0)],
            )
        ]
        mocker.patch("otx.algorithms.visual_prompting.tasks.openvino.OpenvinoAdapter")
        mocker.patch.object(Model, "create_model")
        mocker.patch.object(
            VisualPromptingToAnnotationConverter, "convert_to_annotation", return_value=self.fake_annotation
        )
        self.task_environment = init_environment()
        visual_prompting_hparams = self.task_environment.get_hyper_parameters(VisualPromptingBaseConfig)
        label_schema = self.task_environment.label_schema

        self.visual_prompting_ov_inferencer = OpenVINOVisualPromptingInferencer(
            visual_prompting_hparams,
            label_schema,
            {"image_encoder": "", "decoder": ""},
            {"image_encoder": "", "decoder": ""},
        )
        self.visual_prompting_ov_inferencer.model["decoder"] = mocker.patch(
            "openvino.model_api.models.Model", autospec=True
        )

    @e2e_pytest_unit
    def test_pre_process(self, mocker):
        """Test pre_process."""
        mocker_get_prompts = mocker.patch.object(OTXVisualPromptingDataset, "get_prompts", return_value={})
        mocker.patch.object(self.visual_prompting_ov_inferencer, "transform", lambda items: items)
        mocker.patch.object(
            self.visual_prompting_ov_inferencer.model["image_encoder"], "preprocess", return_value=({}, {})
        )
        mocker.patch.object(self.visual_prompting_ov_inferencer.model["decoder"], "preprocess", return_value=[{}])
        fake_input = mocker.Mock(spec=DatasetItemEntity)

        returned_value = self.visual_prompting_ov_inferencer.pre_process(fake_input)

        assert isinstance(returned_value, tuple)
        mocker_get_prompts.assert_called_once()

    @e2e_pytest_unit
    def test_post_process(self, mocker):
        """Test post_process."""
        fake_prediction = {"masks": np.empty((1, 1, 2, 2))}
        fake_metadata = {"label": mocker.Mock(spec=LabelEntity), "original_size": np.array((2, 2))}
        self.visual_prompting_ov_inferencer.model["decoder"].postprocess.return_value = (
            np.ones((2, 2)),
            np.ones((2, 2)),
        )

        returned_value = self.visual_prompting_ov_inferencer.post_process(fake_prediction, fake_metadata)

        assert len(returned_value) == 3
        assert np.array_equal(returned_value[0], self.fake_annotation)
        assert np.array_equal(returned_value[1], np.ones((2, 2)))
        assert np.array_equal(returned_value[2], np.ones((2, 2)))

    @e2e_pytest_unit
    def test_predict(self, mocker):
        """Teset predict."""
        mocker_pre_process = mocker.patch.object(
            OpenVINOVisualPromptingInferencer,
            "pre_process",
            return_value=(
                torch.zeros((1, 3, 2, 2)),
                {},
                [
                    {
                        "point_coords": [np.array([[[1, 1], [2, 2]]])],
                        "point_labels": [1, 2],
                        "label": LabelEntity(name="fake", domain="VISUALPROMPTING"),
                        "orig_size": (4, 4),
                    }
                ],
            ),
        )
        mocker_forward = mocker.patch.object(
            OpenVINOVisualPromptingInferencer,
            "forward_image_encoder",
            return_value={"image_embeddings": np.empty((4, 2, 2))},
        )
        mocker_forward_decoder = mocker.patch.object(
            OpenVINOVisualPromptingInferencer, "forward_decoder", return_value={"iou_predictions": 0.1}
        )
        mocker_post_process = mocker.patch.object(
            OpenVINOVisualPromptingInferencer, "post_process", return_value=(self.fake_annotation, None, None)
        )
        fake_input = mocker.Mock(spec=DatasetItemEntity)

        returned_value = self.visual_prompting_ov_inferencer.predict(fake_input)

        mocker_pre_process.assert_called_once()
        mocker_forward.assert_called_once()
        mocker_forward_decoder.assert_called_once()
        mocker_post_process.assert_called_once()
        assert returned_value == self.fake_annotation

    @e2e_pytest_unit
    def test_forward_image_encoder(self):
        """Test forward_image_encoder."""
        fake_input = {"images": np.ones((1, 3, 2, 2))}
        fake_output = {"image_embeddings": np.ones((1, 1, 2, 2))}
        self.visual_prompting_ov_inferencer.model["image_encoder"].infer_sync.return_value = fake_output
        returned_value = self.visual_prompting_ov_inferencer.forward_image_encoder(fake_input)

        assert returned_value == fake_output

    @e2e_pytest_unit
    def test_forward_decoder(self):
        """Test forward_decoder."""
        fake_input = {}
        fake_output = {"masks": np.ones((1, 1, 2, 2))}
        self.visual_prompting_ov_inferencer.model["decoder"].infer_sync.return_value = fake_output
        returned_value = self.visual_prompting_ov_inferencer.forward_decoder(fake_input)

        assert returned_value == fake_output


class TestOpenVINOZeroShotVisualPromptingInferencer:
    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        self.fake_annotation = [
            Annotation(
                Polygon(points=[Point(0, 0)]),
                id=0,
                labels=[ScoredLabel(LabelEntity(name="fake", domain="VISUALPROMPTING"), probability=1.0)],
            )
        ]
        mocker.patch("otx.algorithms.visual_prompting.tasks.openvino.OpenvinoAdapter")
        mocker.patch.object(Model, "create_model")
        mocker.patch.object(
            VisualPromptingToAnnotationConverter, "convert_to_annotation", return_value=self.fake_annotation
        )
        self.task_environment = init_environment()
        visual_prompting_hparams = self.task_environment.get_hyper_parameters(VisualPromptingBaseConfig)
        label_schema = self.task_environment.label_schema

        self.zero_shot_visual_prompting_ov_inferencer = OpenVINOZeroShotVisualPromptingInferencer(
            visual_prompting_hparams,
            label_schema,
            {"image_encoder": "", "decoder": ""},
            {"image_encoder": "", "decoder": ""},
        )
        self.zero_shot_visual_prompting_ov_inferencer.model["decoder"] = mocker.patch(
            "otx.algorithms.visual_prompting.tasks.openvino.model_wrappers.Decoder",
            autospec=True,
        )
        self.zero_shot_visual_prompting_ov_inferencer.model["decoder"].mask_threshold = 0.3
        self.zero_shot_visual_prompting_ov_inferencer.model["decoder"]._apply_coords.return_value = np.array([[1, 1]])
        self.zero_shot_visual_prompting_ov_inferencer.model["decoder"].output_blob_name = "upscaled_masks"

    @e2e_pytest_unit
    def test_learn(self, mocker):
        """Test learn."""
        mocker_pre_process = mocker.patch.object(
            OpenVINOVisualPromptingInferencer,
            "pre_process",
            return_value=(
                torch.zeros((1, 3, 2, 2)),
                {"original_shape": np.array((4, 4))},
                [
                    {
                        "point_coords": [np.array([[[1, 1], [2, 2]]])],
                        "point_labels": [1, 2],
                        "label": MockScoredLabel(label=0, name="fake"),
                        "orig_size": (4, 4),
                    }
                ],
            ),
        )
        mocker_forward_image_encoder = mocker.patch.object(
            OpenVINOZeroShotVisualPromptingInferencer,
            "forward_image_encoder",
            return_value={"image_embeddings": np.empty((4, 2, 2))},
        )
        mocker_generate_masked_features = mocker.patch.object(
            OpenVINOZeroShotVisualPromptingInferencer, "_generate_masked_features", return_value=torch.ones(1, 256)
        )

        self.zero_shot_visual_prompting_ov_inferencer.model["decoder"].infer_sync.return_value = {
            "upscaled_masks": np.ones((1, 4, 4, 4), dtype=np.bool),
            "iou_predictions": np.array([[0.9, 0.7, 0.9, 0.8]]),
            "low_res_masks": np.ones((1, 4, 2, 2)),
        }
        mocker_pickle_dump = mocker.patch("otx.algorithms.visual_prompting.tasks.openvino.pickle.dump")
        mocker.patch("builtins.open", return_value="Mocked data")

        fake_input = mocker.Mock(spec=DatasetItemEntity)
        results = self.zero_shot_visual_prompting_ov_inferencer.learn(fake_input, reset_feat=True)

        assert results[0]["reference_feats"].shape == (1, 1, 256)
        assert results[0]["used_indices"] == np.array([[0]])
        assert np.all(results[1] == np.ones((1, 4, 4)))
        mocker_pre_process.assert_called_once()
        mocker_forward_image_encoder.assert_called_once()
        mocker_generate_masked_features.assert_called_once()
        mocker_pickle_dump.assert_called_once()

    @e2e_pytest_unit
    def test_predict(self, mocker):
        """Test predict."""
        mocker_pre_process = mocker.patch.object(
            OpenVINOZeroShotVisualPromptingInferencer,
            "pre_process_image_encoder",
            return_value=(torch.zeros((1, 3, 2, 2)), {"original_shape": (4, 4, 1)}),
        )
        mocker_forward = mocker.patch.object(
            OpenVINOZeroShotVisualPromptingInferencer,
            "forward_image_encoder",
            return_value={"image_embeddings": np.empty((4, 2, 2))},
        )
        mocker_get_prompt_candidates = mocker.patch.object(
            OpenVINOZeroShotVisualPromptingInferencer,
            "_get_prompt_candidates",
            return_value=({0: np.array([[1, 1, 1]])}, {0: np.array([[2, 2]])}),
        )
        mocker_forward_decoder = mocker.patch.object(
            OpenVINOZeroShotVisualPromptingInferencer, "forward_decoder", return_value={"upscaled_masks": None}
        )
        mocker_post_process = mocker.patch.object(
            OpenVINOZeroShotVisualPromptingInferencer, "post_process", return_value=(self.fake_annotation, None, None)
        )
        self.zero_shot_visual_prompting_ov_inferencer.reference_feats = np.random.rand(1, 1, 1)
        self.zero_shot_visual_prompting_ov_inferencer.used_indices = np.array([[0]])
        fake_input = mocker.Mock(spec=DatasetItemEntity)

        results = self.zero_shot_visual_prompting_ov_inferencer.predict(fake_input)

        mocker_pre_process.assert_called_once()
        mocker_forward.assert_called_once()
        mocker_get_prompt_candidates.assert_called_once()
        mocker_forward_decoder.assert_called_once()
        mocker_post_process.assert_called_once()
        assert results == self.fake_annotation

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "postprocess_output,infer_sync_output,expected",
        [
            (
                (np.ones((1, 1)), np.ones((3, 3))),
                {
                    "upscaled_masks": np.ones((3, 3)),
                    "iou_predictions": np.array([[0.9]]),
                    "low_res_masks": np.ones((1, 1, 2, 2)),
                },
                {"upscaled_masks": np.ones((3, 3))},
            ),
            (
                (np.zeros((2, 2)), np.zeros((3, 3))),
                {
                    "upscaled_masks": np.zeros((3, 3)),
                    "iou_predictions": np.array([[0.9]]),
                    "low_res_masks": np.ones((1, 1, 2, 2)),
                },
                {"upscaled_masks": np.zeros((3, 3))},
            ),
        ],
    )
    def test_forward_decoder(
        self,
        mocker,
        postprocess_output: Tuple[torch.Tensor, torch.Tensor],
        infer_sync_output: Dict[str, np.ndarray],
        expected: Dict[str, torch.Tensor],
    ):
        """Test forward_decoder."""
        mocker.patch.object(
            self.zero_shot_visual_prompting_ov_inferencer.model["decoder"], "infer_sync", return_value=infer_sync_output
        )
        mocker.patch.object(
            self.zero_shot_visual_prompting_ov_inferencer.model["decoder"],
            "_apply_coords",
            return_value=np.array([[[1, 1]]], dtype=np.float32),
        )
        mocker.patch.object(
            self.zero_shot_visual_prompting_ov_inferencer, "_postprocess_masks", return_value=postprocess_output
        )

        result = self.zero_shot_visual_prompting_ov_inferencer.forward_decoder(
            inputs={
                "image_embeddings": np.empty((1, 4, 2, 2)),
                "point_coords": np.array([[[1, 1]]], dtype=np.float32),
                "point_labels": np.array([[1]], dtype=np.float32),
            },
            original_size=np.array([3, 3]),
        )

        assert np.all(result["upscaled_masks"] == expected["upscaled_masks"])

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "result_point_selection",
        [np.array([[2, 2, 0.9], [1, 2, 0.8], [0, 2, 0.7], [2, 1, 0.6]]), np.array([[-1, -1, -1]])],
    )
    def test_get_prompt_candidates(self, mocker, result_point_selection: np.ndarray) -> None:
        """Test _get_prompt_candidates."""
        mocker.patch.object(
            OpenVINOZeroShotVisualPromptingInferencer,
            "_point_selection",
            return_value=(result_point_selection, np.zeros((1, 2))),
        )
        image_embeddings = np.ones((1, 4, 4, 4))
        reference_feats = np.random.rand(1, 1, 4)
        used_indices = np.array([0])
        original_shape = np.array([4, 4], dtype=np.int64)

        total_points_scores, total_bg_coords = self.zero_shot_visual_prompting_ov_inferencer._get_prompt_candidates(
            image_embeddings=image_embeddings,
            reference_feats=reference_feats,
            used_indices=used_indices,
            original_shape=original_shape,
            image_size=4,
            downsizing=1,
        )

        assert total_points_scores[0].shape[0] == len(result_point_selection)
        assert total_bg_coords[0].shape[0] == 1

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "mask_sim,expected",
        [
            (
                np.arange(0.1, 1.0, 0.1).reshape(3, 3),
                np.array([[2, 2, 0.9], [1, 2, 0.8], [0, 2, 0.7], [2, 1, 0.6]]),
            ),
            (np.zeros((3, 3)), None),
        ],
    )
    def test_point_selection(self, mask_sim: np.ndarray, expected: np.ndarray) -> None:
        """Test _point_selection."""
        points_scores, bg_coords = self.zero_shot_visual_prompting_ov_inferencer._point_selection(
            mask_sim=mask_sim,
            original_shape=np.array([4, 4]),
            threshold=0.5,
            image_size=4,
            downsizing=1,
        )

        if points_scores is not None:
            assert np.allclose(points_scores, expected)
        else:
            assert points_scores == expected

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "masks,expected_masks",
        [
            (
                np.repeat(np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])[None], 4, axis=0)[None],
                np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.bool_),
            ),
            (
                np.concatenate(
                    (
                        np.repeat(np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])[None], 3, axis=0)[None],
                        np.zeros((1, 1, 3, 3)),
                    ),
                    axis=1,
                ),
                np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.bool_),
            ),
            (np.zeros((1, 4, 3, 3)), np.zeros((3, 3))),
        ],
    )
    def test_postprocess_masks(self, masks: np.ndarray, expected_masks: np.ndarray):
        """Test _postprocess_masks."""
        self.zero_shot_visual_prompting_ov_inferencer.model["decoder"].mask_threshold = 0.0
        self.zero_shot_visual_prompting_ov_inferencer.model["decoder"].image_size = 3

        _, result_masks = self.zero_shot_visual_prompting_ov_inferencer._postprocess_masks(
            masks=masks, logits=np.empty((1, 4, 2, 2)), scores=np.array([[0.5, 0.7, 0.8, 0.9]])
        )

        assert result_masks.shape == (3, 3)
        assert np.all(result_masks == expected_masks)

    @e2e_pytest_unit
    def test_inspect_overlapping_areas(self) -> None:
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

        self.zero_shot_visual_prompting_ov_inferencer._inspect_overlapping_areas(
            predicted_masks, used_points, threshold_iou=0.5
        )

        assert len(predicted_masks[0]) == 2
        assert len(predicted_masks[1]) == 3
        assert all(np.array([2, 2, 0.5]) == used_points[0][0])
        assert all(np.array([0, 0, 0.7]) == used_points[1][2])

    @e2e_pytest_unit
    def test_find_latest_reference_info(self, mocker):
        """Test _find_latest_reference_info."""
        mocker.patch("otx.algorithms.visual_prompting.tasks.openvino.os.path.isdir", return_value=True)

        # there are some saved reference info
        mocker.patch("otx.algorithms.visual_prompting.tasks.openvino.os.listdir", return_value=["1", "2"])
        results = self.zero_shot_visual_prompting_ov_inferencer._find_latest_reference_info()
        assert results == "2"

        # there are no saved reference info
        mocker.patch("otx.algorithms.visual_prompting.tasks.openvino.os.listdir", return_value=[])
        results = self.zero_shot_visual_prompting_ov_inferencer._find_latest_reference_info()
        assert results is None

    @e2e_pytest_unit
    def test_get_reference_info(self, mocker):
        """Test _get_reference_info."""
        mocker.patch("otx.algorithms.visual_prompting.tasks.openvino.os.path.isdir", return_value=True)

        # get previously saved reference info
        mocker.patch("otx.algorithms.visual_prompting.tasks.openvino.os.listdir", return_value=["1", "2"])
        mocker.patch(
            "otx.algorithms.visual_prompting.tasks.openvino.pickle.load",
            return_value={"reference_feats": 1, "used_indices": 2},
        )
        mocker.patch("builtins.open", return_value="Mocked data")

        results = self.zero_shot_visual_prompting_ov_inferencer._get_reference_info()
        assert results == (1, 2)

        # no saved reference info
        mocker.patch("otx.algorithms.visual_prompting.tasks.openvino.os.listdir", return_value=[])

        results = self.zero_shot_visual_prompting_ov_inferencer._get_reference_info()
        assert results == (None, None)

    @e2e_pytest_unit
    def test_expand_reference_info(self):
        """Test expand_reference_info."""
        self.zero_shot_visual_prompting_ov_inferencer.reference_feats = np.ones((3, 2, 2))
        new_largest_label = 5

        self.zero_shot_visual_prompting_ov_inferencer.expand_reference_info(new_largest_label)

        assert self.zero_shot_visual_prompting_ov_inferencer.reference_feats.shape == (6, 2, 2)
        assert np.all(self.zero_shot_visual_prompting_ov_inferencer.reference_feats[:3] == 1.0)
        assert np.all(self.zero_shot_visual_prompting_ov_inferencer.reference_feats[3:] == 0.0)

    @e2e_pytest_unit
    def test_generate_masked_features(self) -> None:
        """Test _generate_masked_features."""
        self.zero_shot_visual_prompting_ov_inferencer.model["image_encoder"].image_size = 16
        feats = np.random.rand(8, 8, 1)
        masks = np.zeros((16, 16), dtype=np.float32)
        masks[4:12, 4:12] = 1.0

        masked_feat = self.zero_shot_visual_prompting_ov_inferencer._generate_masked_features(
            feats=feats, masks=masks, threshold_mask=0.3
        )

        assert masked_feat.shape == (1, 1)

    @e2e_pytest_unit
    def test_pad_to_square(self) -> None:
        """Test _pad_to_square."""
        self.zero_shot_visual_prompting_ov_inferencer.model["image_encoder"].image_size = 16

        result = self.zero_shot_visual_prompting_ov_inferencer._pad_to_square(x=np.ones((8, 8)))

        assert result[:8, :8].sum() == 8**2
        assert result[:8, 8:].sum() == 0
        assert result[8:, :8].sum() == 0
        assert result[8:, 8:].sum() == 0


class TestOTXOpenVinoDataLoader:
    @pytest.fixture
    def load_dataloader(self, mocker):
        def _load_dataloader(module_name: str, output_model: Optional[ModelEntity] = None):
            dataset = generate_visual_prompting_dataset()
            dataset = dataset.get_subset(Subset.TRAINING)
            return OTXOpenVinoDataLoader(dataset, self.mocker_inferencer, module_name, output_model=output_model)

        return _load_dataloader

    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        self.mocker_read_model = mocker.patch("otx.algorithms.visual_prompting.tasks.openvino.ov.Core.read_model")
        self.mocker_compile_model = mocker.patch("otx.algorithms.visual_prompting.tasks.openvino.ov.Core.compile_model")
        self.mocker_inferencer = mocker.patch.object(OpenVINOVisualPromptingInferencer, "__init__")

    @e2e_pytest_unit
    @pytest.mark.parametrize("module_name", ["image_encoder", "decoder"])
    def test_getitem(self, mocker, load_dataloader, module_name: str):
        """Test __getitem__."""
        mocker_output_model = mocker.patch("otx.api.entities.model.ModelEntity")
        if module_name == "decoder":
            mocker.patch.object(mocker_output_model, "get_data")
            self.mocker_read_model.reset_mock()
            self.mocker_compile_model.reset_mock()

        dataloader = load_dataloader(module_name, mocker_output_model)

        setattr(dataloader, "target_length", 8)
        mocker.patch.object(
            dataloader.inferencer,
            "pre_process",
            return_value=({"images": np.zeros((1, 3, 4, 4), dtype=np.uint8)}, None, [{"label": 1, "orig_size": 1}]),
        )

        results = dataloader.__getitem__(0)

        if module_name == "image_encoder":
            assert results["images"].shape == (1, 3, 8, 8)
        else:
            self.mocker_read_model.assert_called_once()
            self.mocker_compile_model.assert_called_once()
            assert "label" not in results
            assert "orig_size" in results
            assert "image_embeddings" in results


class TestOpenVINOVisualPromptingTask:
    @pytest.fixture
    def otx_model(self):
        model_configuration = ModelConfiguration(
            configurable_parameters=ConfigurableParameters(header="header", description="description"),
            label_schema=LabelSchemaEntity(),
        )
        return ModelEntity(train_dataset=DatasetEntity(), configuration=model_configuration)

    @pytest.fixture(autouse=True)
    def setup(self, mocker, otx_model):
        """Load the OpenVINOVisualPromptingTask."""
        mocker.patch("otx.algorithms.visual_prompting.tasks.openvino.OpenvinoAdapter")
        mocker.patch.object(Model, "create_model")
        self.task_environment = init_environment()
        visual_prompting_hparams = self.task_environment.get_hyper_parameters(VisualPromptingBaseConfig)

        visual_prompting_ov_inferencer = OpenVINOVisualPromptingInferencer(
            visual_prompting_hparams,
            self.task_environment.label_schema,
            {"image_encoder": "", "decoder": ""},
            {"image_encoder": "", "decoder": ""},
        )

        # self.task_environment.model = mocker.patch("otx.api.entities.model.ModelEntity")
        self.task_environment.model = otx_model
        mocker.patch.object(OpenVINOVisualPromptingTask, "load_inferencer", return_value=visual_prompting_ov_inferencer)
        self.visual_prompting_ov_task = OpenVINOVisualPromptingTask(task_environment=self.task_environment)

    @e2e_pytest_unit
    def test_infer(self, mocker):
        """Test infer."""
        fake_annotation = [
            Annotation(
                Polygon(points=[Point(0, 0)]),
                id=0,
                labels=[ScoredLabel(LabelEntity(name="fake", domain="VISUALPROMPTING"), probability=1.0)],
            )
        ]

        mocker_predict = mocker.patch.object(OpenVINOVisualPromptingInferencer, "predict", return_value=fake_annotation)
        mocker.patch.object(ShapeFactory, "shape_produces_valid_crop", return_value=True)

        dataset = generate_visual_prompting_dataset()

        updated_dataset = self.visual_prompting_ov_task.infer(
            dataset, InferenceParameters(enable_async_inference=False)
        )

        for updated in updated_dataset:
            assert updated.annotation_scene.contains_any([LabelEntity(name="fake", domain="VISUALPROMPTING")])

        mocker_predict.assert_called()
        assert mocker_predict.call_count == len(updated_dataset)

    @e2e_pytest_unit
    def test_evaluate(self, mocker):
        """Test evaluate."""
        result_set = ResultSetEntity(
            model=None,
            ground_truth_dataset=DatasetEntity(),
            prediction_dataset=DatasetEntity(),
        )
        fake_metrics = mocker.patch("otx.api.usecases.evaluation.dice.DiceAverage", autospec=True)
        fake_metrics.get_performance.return_value = Performance(
            score=ScoreMetric(name="fake", value=0.1), dashboard_metrics="mDice"
        )
        mocker.patch.object(MetricsHelper, "compute_dice_averaged_over_pixels", return_value=fake_metrics)
        self.visual_prompting_ov_task.evaluate(result_set)

        assert result_set.performance.score.value == 0.1

    @e2e_pytest_unit
    def test_deploy(self):
        """Test deploy."""
        output_model = deepcopy(self.task_environment.model)
        self.visual_prompting_ov_task.model.set_data("visual_prompting_image_encoder.xml", b"image_encoder_xml")
        self.visual_prompting_ov_task.model.set_data("visual_prompting_image_encoder.bin", b"image_encoder_bin")
        self.visual_prompting_ov_task.model.set_data("visual_prompting_decoder.xml", b"decoder_xml")
        self.visual_prompting_ov_task.model.set_data("visual_prompting_decoder.bin", b"decoder_bin")

        self.visual_prompting_ov_task.deploy(output_model)

        assert output_model.exportable_code is not None

    @e2e_pytest_unit
    def test_optimize(self, mocker):
        """Test optimize."""

        def patch_save_model(model, output_xml):
            output_bin = output_xml.replace(".xml", ".bin")
            with open(output_xml, "wb") as f:
                f.write(f"compressed_{os.path.basename(output_xml)}".encode("utf-8"))
            with open(output_bin, "wb") as f:
                f.write(f"compressed_{os.path.basename(output_bin)}".encode("utf-8"))

        dataset = generate_visual_prompting_dataset()
        output_model = deepcopy(self.task_environment.model)
        self.visual_prompting_ov_task.model.set_data("visual_prompting_image_encoder.xml", b"image_encoder_xml")
        self.visual_prompting_ov_task.model.set_data("visual_prompting_image_encoder.bin", b"image_encoder_bin")
        self.visual_prompting_ov_task.model.set_data("visual_prompting_decoder.xml", b"decoder_xml")
        self.visual_prompting_ov_task.model.set_data("visual_prompting_decoder.bin", b"decoder_bin")
        mocker.patch("otx.algorithms.visual_prompting.tasks.openvino.ov.Core.read_model", autospec=True)
        mocker.patch("otx.algorithms.visual_prompting.tasks.openvino.ov.save_model", new=patch_save_model)
        mocker.patch("otx.algorithms.visual_prompting.tasks.openvino.ov.Core.compile_model")
        fake_quantize = mocker.patch("otx.algorithms.visual_prompting.tasks.openvino.nncf.quantize", autospec=True)

        self.visual_prompting_ov_task.optimize(OptimizationType.POT, dataset=dataset, output_model=output_model)

        fake_quantize.assert_called()
        assert fake_quantize.call_count == 2

        assert (
            self.visual_prompting_ov_task.model.get_data("visual_prompting_image_encoder.xml")
            == b"compressed_visual_prompting_image_encoder.xml"
        )
        assert (
            self.visual_prompting_ov_task.model.get_data("visual_prompting_image_encoder.bin")
            == b"compressed_visual_prompting_image_encoder.bin"
        )
        assert (
            self.visual_prompting_ov_task.model.get_data("visual_prompting_decoder.xml")
            == b"compressed_visual_prompting_decoder.xml"
        )
        assert (
            self.visual_prompting_ov_task.model.get_data("visual_prompting_decoder.bin")
            == b"compressed_visual_prompting_decoder.bin"
        )


class TestOpenVINOZeroShotVisualPromptingTask:
    @pytest.fixture
    def otx_model(self):
        model_configuration = ModelConfiguration(
            configurable_parameters=ConfigurableParameters(header="header", description="description"),
            label_schema=LabelSchemaEntity(),
        )
        return ModelEntity(train_dataset=DatasetEntity(), configuration=model_configuration)

    @pytest.fixture(autouse=True)
    def setup(self, mocker, otx_model):
        """Load the OpenVINOZeroShotVisualPromptingTask."""
        mocker.patch("otx.algorithms.visual_prompting.tasks.openvino.OpenvinoAdapter")
        mocker.patch.object(Model, "create_model")
        self.task_environment = init_environment()
        visual_prompting_hparams = self.task_environment.get_hyper_parameters(VisualPromptingBaseConfig)

        visual_prompting_ov_inferencer = OpenVINOZeroShotVisualPromptingInferencer(
            visual_prompting_hparams,
            self.task_environment.label_schema,
            {"image_encoder": "", "decoder": ""},
            {"image_encoder": "", "decoder": ""},
        )

        # self.task_environment.model = mocker.patch("otx.api.entities.model.ModelEntity")
        self.task_environment.model = otx_model
        mocker.patch.object(
            OpenVINOZeroShotVisualPromptingTask, "load_inferencer", return_value=visual_prompting_ov_inferencer
        )
        self.zero_shot_visual_prompting_ov_task = OpenVINOZeroShotVisualPromptingTask(
            task_environment=self.task_environment
        )

    @e2e_pytest_unit
    def test_infer_without_reference_info(self):
        """Test infer without reference_info."""
        dataset = generate_visual_prompting_dataset()

        updated_dataset = self.zero_shot_visual_prompting_ov_task.infer(
            dataset, InferenceParameters(enable_async_inference=False), "empty_dir"
        )

        for updated in updated_dataset:
            assert len(updated.annotation_scene.annotations) == 0

    @e2e_pytest_unit
    def test_infer_with_reference_info(self, mocker):
        """Test infer with reference_info."""
        fake_annotation = [
            Annotation(
                Polygon(points=[Point(0, 0)]),
                id=0,
                labels=[ScoredLabel(LabelEntity(name="fake", domain="VISUALPROMPTING"), probability=1.0)],
            )
        ]

        mocker_predict = mocker.patch.object(
            OpenVINOZeroShotVisualPromptingInferencer, "predict", return_value=fake_annotation
        )
        mocker.patch.object(ShapeFactory, "shape_produces_valid_crop", return_value=True)
        mocker.patch.object(
            self.zero_shot_visual_prompting_ov_task.inferencer, "_get_reference_info", return_value=({}, {})
        )

        dataset = generate_visual_prompting_dataset()

        updated_dataset = self.zero_shot_visual_prompting_ov_task.infer(
            dataset, InferenceParameters(enable_async_inference=False)
        )

        for updated in updated_dataset:
            assert updated.annotation_scene.contains_any([LabelEntity(name="fake", domain="VISUALPROMPTING")])

        mocker_predict.assert_called()
        assert mocker_predict.call_count == len(updated_dataset)

    @e2e_pytest_unit
    def test_optimize(self, mocker):
        """Test optimize."""

        def patch_save_model(model, output_xml):
            output_bin = output_xml.replace(".xml", ".bin")
            with open(output_xml, "wb") as f:
                f.write(f"compressed_{os.path.basename(output_xml)}".encode("utf-8"))
            with open(output_bin, "wb") as f:
                f.write(f"compressed_{os.path.basename(output_bin)}".encode("utf-8"))

        dataset = generate_visual_prompting_dataset()
        output_model = deepcopy(self.task_environment.model)
        self.zero_shot_visual_prompting_ov_task.model.set_data(
            "visual_prompting_image_encoder.xml", b"image_encoder_xml"
        )
        self.zero_shot_visual_prompting_ov_task.model.set_data(
            "visual_prompting_image_encoder.bin", b"image_encoder_bin"
        )
        self.zero_shot_visual_prompting_ov_task.model.set_data("visual_prompting_decoder.xml", b"decoder_xml")
        self.zero_shot_visual_prompting_ov_task.model.set_data("visual_prompting_decoder.bin", b"decoder_bin")
        mocker.patch("otx.algorithms.visual_prompting.tasks.openvino.ov.Core.read_model", autospec=True)
        mocker.patch("otx.algorithms.visual_prompting.tasks.openvino.ov.save_model", new=patch_save_model)
        mocker.patch("otx.algorithms.visual_prompting.tasks.openvino.ov.Core.compile_model")
        fake_quantize = mocker.patch("otx.algorithms.visual_prompting.tasks.openvino.nncf.quantize", autospec=True)

        self.zero_shot_visual_prompting_ov_task.optimize(
            OptimizationType.POT, dataset=dataset, output_model=output_model
        )

        fake_quantize.assert_called()
        assert fake_quantize.call_count == 2

        assert (
            self.zero_shot_visual_prompting_ov_task.model.get_data("visual_prompting_image_encoder.xml")
            == b"compressed_visual_prompting_image_encoder.xml"
        )
        assert (
            self.zero_shot_visual_prompting_ov_task.model.get_data("visual_prompting_image_encoder.bin")
            == b"compressed_visual_prompting_image_encoder.bin"
        )
        assert (
            self.zero_shot_visual_prompting_ov_task.model.get_data("visual_prompting_decoder.xml")
            == b"compressed_visual_prompting_decoder.xml"
        )
        assert (
            self.zero_shot_visual_prompting_ov_task.model.get_data("visual_prompting_decoder.bin")
            == b"compressed_visual_prompting_decoder.bin"
        )
