"""Tests the methods in the OpenVINO task."""

# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from otx.algorithms.anomaly.tasks.openvino import OpenVINOTask
from otx.algorithms.anomaly.tasks.train import TrainingTask
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.entities.model import ModelEntity, ModelOptimizationType
from otx.api.entities.model_template import TaskType
from otx.api.entities.optimization_parameters import OptimizationParameters
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from otx.api.usecases.tasks.interfaces.optimization_interface import OptimizationType
from otx.cli.utils.io import read_model


class TestOpenVINOTask:
    """Tests methods in the OpenVINO task."""

    @pytest.fixture
    def tmp_dir(self):
        with TemporaryDirectory() as tmp_dir:
            yield tmp_dir

    def set_normalization_params(self, output_model: ModelEntity):
        """Sets normalization parameters for an untrained output model.

        This is needed as untrained model might have nan values for normalization parameters which will raise an error.
        """
        output_model.set_data("image_threshold", np.float32(0.5).tobytes())
        output_model.set_data("pixel_threshold", np.float32(0.5).tobytes())
        output_model.set_data("min", np.float32(0).tobytes())
        output_model.set_data("max", np.float32(1).tobytes())

    @pytest.mark.skip(reason="CVS-107918 FAIL code -11 in anomaly unit test on python3.10")
    def test_openvino(self, tmpdir, setup_task_environment):
        """Tests the OpenVINO optimize method."""
        root = str(tmpdir.mkdir("anomaly_openvino_test"))

        setup_task_environment = deepcopy(setup_task_environment)  # since fixture is mutable
        task_type = setup_task_environment.task_type
        dataset: DatasetEntity = setup_task_environment.dataset
        task_environment = setup_task_environment.task_environment
        output_model = setup_task_environment.output_model

        # set normalization params for the output model
        train_task = TrainingTask(task_environment, output_path=root)
        self.set_normalization_params(output_model)
        train_task.save_model(output_model)
        task_environment.model = output_model
        train_task.export(ExportType.OPENVINO, output_model)

        # Create OpenVINO task
        openvino_task = OpenVINOTask(task_environment)

        # call inference
        dataset = dataset.get_subset(Subset.VALIDATION)
        predicted_dataset = openvino_task.infer(
            dataset.with_empty_annotations(), InferenceParameters(is_evaluation=True)
        )

        # call evaluate
        result_set = ResultSetEntity(output_model, dataset, predicted_dataset)
        openvino_task.evaluate(result_set)
        if task_type in (TaskType.ANOMALY_CLASSIFICATION, TaskType.ANOMALY_DETECTION):
            assert result_set.performance.score.name == "f-measure"
        elif task_type == TaskType.ANOMALY_SEGMENTATION:
            assert result_set.performance.score.name == "Dice Average"

        # optimize to POT
        openvino_task.optimize(OptimizationType.POT, dataset, output_model, OptimizationParameters())
        assert output_model.optimization_type == ModelOptimizationType.POT
        assert output_model.get_data("label_schema.json") is not None

        # deploy
        openvino_task.deploy(output_model)
        assert output_model.exportable_code is not None

    @patch.multiple(OpenVINOTask, get_config=MagicMock(), load_inferencer=MagicMock())
    @patch("otx.algorithms.anomaly.tasks.openvino.get_transforms", MagicMock())
    def test_anomaly_legacy_keys(self, mocker, tmp_dir):
        """Checks whether the model is loaded correctly with legacy and current keys."""

        tmp_dir = Path(tmp_dir)
        xml_model_path = tmp_dir / "model.xml"
        xml_model_path.write_text("xml_model")
        bin_model_path = tmp_dir / "model.bin"
        bin_model_path.write_text("bin_model")

        # Test loading legacy keys
        legacy_keys = ("image_threshold", "pixel_threshold", "min", "max")
        for key in legacy_keys:
            (tmp_dir / key).write_bytes(np.zeros(1, dtype=np.float32).tobytes())

        model = read_model(mocker.MagicMock(), str(xml_model_path), mocker.MagicMock())
        task_environment = TaskEnvironment(
            model_template=mocker.MagicMock(),
            model=model,
            hyper_parameters=mocker.MagicMock(),
            label_schema=LabelSchemaEntity.from_labels(
                [
                    LabelEntity("Anomalous", is_anomalous=True, domain=Domain.ANOMALY_SEGMENTATION),
                    LabelEntity("Normal", domain=Domain.ANOMALY_SEGMENTATION),
                ]
            ),
        )
        openvino_task = OpenVINOTask(task_environment)
        metadata = openvino_task.get_metadata()
        for key in legacy_keys:
            assert metadata[key] == np.zeros(1, dtype=np.float32)

        # cleanup legacy keys
        for key in legacy_keys:
            (tmp_dir / key).unlink()

        # Test loading new keys
        new_metadata = {
            "image_threshold": np.zeros(1, dtype=np.float32).tolist(),
            "pixel_threshold": np.zeros(1, dtype=np.float32).tolist(),
            "min": np.zeros(1, dtype=np.float32).tolist(),
            "max": np.zeros(1, dtype=np.float32).tolist(),
        }
        (tmp_dir / "metadata").write_bytes(json.dumps(new_metadata).encode())
        task_environment.model = read_model(mocker.MagicMock(), str(xml_model_path), mocker.MagicMock())
        openvino_task = OpenVINOTask(task_environment)
        metadata = openvino_task.get_metadata()
        for key in new_metadata.keys():
            assert metadata[key] == np.zeros(1, dtype=np.float32)
