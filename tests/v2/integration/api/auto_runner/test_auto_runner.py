# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from otx.v2.api.core import AutoRunner

from tests.v2.integration.api.test_helper import TASK_CONFIGURATION


class TestAutoRunnerAPI:
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
            task=task,
            train_data_roots=task_configuration["train_data_roots"],
            val_data_roots=task_configuration["val_data_roots"],
            test_data_roots=task_configuration["test_data_roots"],
        )

        results = auto_runner.train(
            max_epochs=1,
        )
        assert "model" in results
        assert "checkpoint" in  results
        assert isinstance(results["checkpoint"], str)
        assert Path(results["checkpoint"]).exists()

        # Validation
        val_score = auto_runner.validate()
        # TODO: Check Validation Results

        # Test
        test_score = auto_runner.test()
        # TODO: Check Testing Results

        # Prediction with single image
        pred_result = auto_runner.predict(
            model=results["model"],
            checkpoint=results["checkpoint"],
            img=task_configuration["sample"],
        )
        # TODO: Check Prediction Results

        # Export Openvino IR Model
        export_output = auto_runner.export(
            model=results["model"],
            checkpoint=results["checkpoint"],
        )
        assert isinstance(export_output, dict)
        assert "outputs" in export_output
        assert isinstance(export_output["outputs"], dict)
        assert "bin" in export_output["outputs"]
        assert "xml" in export_output["outputs"]
        assert Path(export_output["outputs"]["bin"]).exists()
        assert Path(export_output["outputs"]["xml"]).exists()
