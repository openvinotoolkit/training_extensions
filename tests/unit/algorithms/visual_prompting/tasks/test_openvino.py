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
    OTXZeroShotOpenVinoDataLoader,
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
            OpenVINOVisualPromptingInferencer, "forward_decoder", return_value=None
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

        self.visual_prompting_ov_inferencer = OpenVINOZeroShotVisualPromptingInferencer(
            visual_prompting_hparams,
            label_schema,
            {"image_encoder": "", "prompt_getter": "", "decoder": ""},
            {"image_encoder": "", "prompt_getter": "", "decoder": ""},
        )
        self.visual_prompting_ov_inferencer.model["decoder"] = mocker.patch(
            "otx.algorithms.visual_prompting.tasks.openvino.model_wrappers.Decoder", autospec=True
        )
        self.visual_prompting_ov_inferencer.model["decoder"]._apply_coords.return_value = np.array([[1, 1]])

    @e2e_pytest_unit
    def test_predict(self, mocker):
        """Test predict."""
        mocker_pre_process = mocker.patch.object(
            OpenVINOZeroShotVisualPromptingInferencer,
            "pre_process",
            return_value=(torch.zeros((1, 3, 2, 2)), {"original_shape": (4, 4, 1)}),
        )
        mocker_forward = mocker.patch.object(
            OpenVINOZeroShotVisualPromptingInferencer,
            "forward_image_encoder",
            return_value={"image_embeddings": np.empty((4, 2, 2))},
        )
        mocker_forward_decoder = mocker.patch.object(
            OpenVINOZeroShotVisualPromptingInferencer,
            "forward_prompt_getter",
            return_value={"total_points_scores": np.array([[[1, 1, 1]]]), "total_bg_coords": np.array([[[2, 2]]])},
        )
        mocker_forward_decoder = mocker.patch.object(
            OpenVINOZeroShotVisualPromptingInferencer, "forward_decoder", return_value=None
        )
        mocker_post_process = mocker.patch.object(
            OpenVINOZeroShotVisualPromptingInferencer, "post_process", return_value=(self.fake_annotation, None, None)
        )
        fake_input = mocker.Mock(spec=DatasetItemEntity)

        returned_value = self.visual_prompting_ov_inferencer.predict(fake_input)

        mocker_pre_process.assert_called_once()
        mocker_forward.assert_called_once()
        mocker_forward_decoder.assert_called_once()
        mocker_post_process.assert_called_once()
        assert returned_value == self.fake_annotation

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "postprocess_output,infer_sync_output,expected",
        [
            (
                (np.ones((1, 1)), np.ones((3, 3)), 0.9),
                {"iou_predictions": np.array([[0.9]]), "low_res_masks": np.ones((1, 1, 2, 2))},
                {"iou_predictions": np.array([[0.9]]), "low_res_masks": np.ones((1, 1, 2, 2))},
            ),
            (
                (np.zeros((2, 2)), np.zeros((3, 3)), 0.0),
                {"iou_predictions": np.array([[0.9]]), "low_res_masks": np.ones((1, 1, 2, 2))},
                {"iou_predictions": 0.0, "low_res_masks": np.zeros((2, 2))},
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
            self.visual_prompting_ov_inferencer.model["decoder"], "infer_sync", return_value=infer_sync_output
        )
        mocker.patch.object(
            self.visual_prompting_ov_inferencer.model["decoder"],
            "_apply_coords",
            return_value=np.array([[[1, 1]]], dtype=np.float32),
        )
        mocker.patch.object(self.visual_prompting_ov_inferencer, "_postprocess_masks", return_value=postprocess_output)

        result = self.visual_prompting_ov_inferencer.forward_decoder(
            inputs={
                "image_embeddings": np.empty((1, 4, 2, 2)),
                "point_coords": np.array([[[1, 1]]], dtype=np.float32),
                "point_labels": np.array([[1]], dtype=np.float32),
            },
            original_size=np.array([3, 3]),
        )

        assert np.all(result["iou_predictions"] == expected["iou_predictions"])
        assert np.all(result["low_res_masks"] == expected["low_res_masks"])

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "high_res_masks,expected_masks,expected_scores",
        [
            (
                np.repeat(np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])[..., None], 4, axis=-1),
                np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.bool_),
                0.9,
            ),
            (
                np.concatenate(
                    (
                        np.repeat(np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])[..., None], 3, axis=-1),
                        np.zeros((3, 3, 1)),
                    ),
                    axis=-1,
                ),
                np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.bool_),
                0.8,
            ),
            (np.zeros((3, 3, 4)), np.zeros((3, 3)), 0.0),
        ],
    )
    def test_postprocess_masks(self, high_res_masks: np.ndarray, expected_masks: np.ndarray, expected_scores: float):
        """Test _postprocess_masks."""
        self.visual_prompting_ov_inferencer.model["decoder"].resize_and_crop.return_value = high_res_masks
        self.visual_prompting_ov_inferencer.model["decoder"].mask_threshold = 0.0
        self.visual_prompting_ov_inferencer.model["decoder"].image_size = 3

        _, result_masks, result_scores = self.visual_prompting_ov_inferencer._postprocess_masks(
            logits=np.empty((1, 4, 2, 2)), scores=np.array([[0.5, 0.7, 0.8, 0.9]]), original_size=np.array([3, 3])
        )

        assert result_masks.shape == (3, 3)
        assert np.all(result_masks == expected_masks)
        assert result_scores == expected_scores


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


class TestOTXZeroShotOpenVinoDataLoader:
    @pytest.fixture
    def load_dataloader(self, mocker):
        def _load_dataloader(module_name: str, output_model: Optional[ModelEntity] = None):
            dataset = generate_visual_prompting_dataset()
            dataset = dataset.get_subset(Subset.TRAINING)
            return OTXZeroShotOpenVinoDataLoader(
                dataset, self.mocker_inferencer, module_name, output_model=output_model
            )

        return _load_dataloader

    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        self.mocker_read_model = mocker.patch("otx.algorithms.visual_prompting.tasks.openvino.ov.Core.read_model")
        self.mocker_compile_model = mocker.patch("otx.algorithms.visual_prompting.tasks.openvino.ov.Core.compile_model")
        self.mocker_inferencer = mocker.patch.object(OpenVINOZeroShotVisualPromptingTask, "__init__")

    @e2e_pytest_unit
    @pytest.mark.parametrize("module_name", ["image_encoder", "prompt_getter", "decoder"])
    def test_getitem(self, mocker, load_dataloader, module_name: str):
        """Test __getitem__."""
        mocker_output_model = mocker.patch("otx.api.entities.model.ModelEntity")
        if module_name in ["prompt_getter", "decoder"]:
            mocker.patch.object(mocker_output_model, "get_data")
            self.mocker_read_model.reset_mock()
            self.mocker_compile_model.reset_mock()

        dataloader = load_dataloader(module_name, mocker_output_model)

        setattr(dataloader, "target_length", 8)
        mocker.patch.object(
            dataloader.inferencer,
            "pre_process",
            return_value=({"images": np.zeros((1, 3, 4, 4), dtype=np.uint8)}, {"original_shape": (4, 4)}),
        )
        if module_name == "decoder":
            mocker.patch.object(
                dataloader,
                "prompt_getter",
                return_value={
                    "total_points_scores": [np.array([[0, 0, 0.5]])],
                    "total_bg_coords": [np.array([[1, 1]])],
                },
            )

        results = dataloader.__getitem__(0)

        if module_name == "image_encoder":
            assert results["images"].shape == (1, 3, 8, 8)
        elif module_name == "prompt_getter":
            self.mocker_read_model.assert_called_once()
            self.mocker_compile_model.assert_called_once()
        else:  # decoder
            self.mocker_read_model.call_count == 2
            self.mocker_compile_model.call_count == 2


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
        mocker.patch("otx.algorithms.visual_prompting.tasks.openvino.ov.serialize", new=patch_save_model)
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
            {"image_encoder": "", "prompt_getter": "", "decoder": ""},
            {"image_encoder": "", "prompt_getter": "", "decoder": ""},
        )

        # self.task_environment.model = mocker.patch("otx.api.entities.model.ModelEntity")
        self.task_environment.model = otx_model
        mocker.patch.object(
            OpenVINOZeroShotVisualPromptingTask, "load_inferencer", return_value=visual_prompting_ov_inferencer
        )
        self.visual_prompting_ov_task = OpenVINOZeroShotVisualPromptingTask(task_environment=self.task_environment)

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
        self.visual_prompting_ov_task.model.set_data("visual_prompting_prompt_getter.xml", b"prompt_getter_xml")
        self.visual_prompting_ov_task.model.set_data("visual_prompting_prompt_getter.bin", b"prompt_getter_bin")
        self.visual_prompting_ov_task.model.set_data("visual_prompting_decoder.xml", b"decoder_xml")
        self.visual_prompting_ov_task.model.set_data("visual_prompting_decoder.bin", b"decoder_bin")
        mocker.patch("otx.algorithms.visual_prompting.tasks.openvino.ov.Core.read_model", autospec=True)
        mocker.patch("otx.algorithms.visual_prompting.tasks.openvino.ov.serialize", new=patch_save_model)
        mocker.patch("otx.algorithms.visual_prompting.tasks.openvino.ov.Core.compile_model")
        fake_quantize = mocker.patch("otx.algorithms.visual_prompting.tasks.openvino.nncf.quantize", autospec=True)

        self.visual_prompting_ov_task.optimize(OptimizationType.POT, dataset=dataset, output_model=output_model)

        fake_quantize.assert_called()
        assert fake_quantize.call_count == 3

        assert (
            self.visual_prompting_ov_task.model.get_data("visual_prompting_image_encoder.xml")
            == b"compressed_visual_prompting_image_encoder.xml"
        )
        assert (
            self.visual_prompting_ov_task.model.get_data("visual_prompting_image_encoder.bin")
            == b"compressed_visual_prompting_image_encoder.bin"
        )
        assert (
            self.visual_prompting_ov_task.model.get_data("visual_prompting_prompt_getter.xml")
            == b"compressed_visual_prompting_prompt_getter.xml"
        )
        assert (
            self.visual_prompting_ov_task.model.get_data("visual_prompting_prompt_getter.bin")
            == b"compressed_visual_prompting_prompt_getter.bin"
        )
        assert (
            self.visual_prompting_ov_task.model.get_data("visual_prompting_decoder.xml")
            == b"compressed_visual_prompting_decoder.xml"
        )
        assert (
            self.visual_prompting_ov_task.model.get_data("visual_prompting_decoder.bin")
            == b"compressed_visual_prompting_decoder.bin"
        )
