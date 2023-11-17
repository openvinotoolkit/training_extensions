# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from otx.v2.api.core import AutoRunner
from otx.v2.api.entities.task_type import TaskType

from tests.v2.integration.test_helper import TASK_CONFIGURATION

# NOTE: This test currently only checks the basic pipeline and doesn't do much checking on the result, which will be fixed in the future.

class TestAutoRunnerAPI:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.task_type_dict = {
            "classification": TaskType.CLASSIFICATION,
            "anomaly_classification": TaskType.ANOMALY_CLASSIFICATION,
            "visual_prompting": TaskType.VISUAL_PROMPTING,
            "segmentation": TaskType.SEGMENTATION,
            "detection": TaskType.DETECTION,
        }

    @pytest.mark.parametrize("task", TASK_CONFIGURATION.keys())
    def test_auto_training_api(self, task: str, tmp_dir_path: Path) -> None:
        """
        Test the AutoRunner API by training, validating, testing, and exporting a model.

        This checks only one model for each task.

        Args:
            dataset (Dataset): The dataset to use for training, validation, and testing.
            model (str): The name of the model to use.
            tmp_dir_path (Path): The path to the temporary directory to use for storing checkpoints and exported models.

        Steps:
        1. Setup Engine
        3. Training (1 epochs)
        4. Validation & Testing
        5. Prediction with single image
        6. Export Openvino IR Model

        Returns:
            None
        """
        # Setup Engine
        task_configuration = TASK_CONFIGURATION[task]
        auto_runner = AutoRunner(
            work_dir=tmp_dir_path,
            task=self.task_type_dict[task],
            train_data_roots=task_configuration["train_data_roots"],
            val_data_roots=task_configuration["val_data_roots"],
            test_data_roots=task_configuration["test_data_roots"],
        )

        results = auto_runner.train(
            max_epochs=1,
        )
        assert "model" in results
        assert "checkpoint" in results
        assert isinstance(results["checkpoint"], str)
        assert Path(results["checkpoint"]).exists()

        # Validation
        auto_runner.validate()

        # Test
        auto_runner.test()

        # Prediction with single image
        auto_runner.predict(
            model=results["model"],
            checkpoint=results["checkpoint"],
            img=task_configuration["sample"],
        )

        # Export Openvino IR Model
        export_output = auto_runner.export(
            checkpoint=results["checkpoint"],
        )
        assert isinstance(export_output, dict)
        assert "outputs" in export_output
        assert isinstance(export_output["outputs"], dict)
        assert "onnx" in export_output["outputs"]
