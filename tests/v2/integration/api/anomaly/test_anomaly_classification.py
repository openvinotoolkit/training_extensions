# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path

import pytest
import torch
from otx.v2.adapters.torch.lightning.anomalib import Dataset, Engine, get_model, list_models

from tests.v2.integration.test_helper import TASK_CONFIGURATION

# Test-related datasets are managed by tests.v2.integration.api.test_helper.
PREDICTION_SAMPLE = TASK_CONFIGURATION["anomaly_classification"]["sample"]

MODELS: list = list_models("otx*")

class TestAnomalibClassificationAPI:
    """
    This class contains integration tests for the training API of the Anomalib library.
    It tests the dataset API, model API, and engine API for pretraining a model on a given dataset.
    """
    @pytest.fixture()
    def dataset(self) -> Dataset:
            """
            Returns a Dataset object containing the paths to the training, validation, and test data.

            Returns:
                Dataset: A Dataset object containing the paths to the training, validation, and test data.
            """
            return Dataset(
                train_data_roots=TASK_CONFIGURATION["anomaly_classification"]["train_data_roots"],
                val_data_roots=TASK_CONFIGURATION["anomaly_classification"]["val_data_roots"],
                test_data_roots=TASK_CONFIGURATION["anomaly_classification"]["test_data_roots"],
            )

    @pytest.mark.parametrize("model", MODELS)
    def test_engine_api(self, dataset: Dataset, model: str, tmp_dir_path: Path) -> None:
        """
        Test the engine API by training, validating, testing, and exporting a model.

        Args:
            dataset (Dataset): The dataset to use for training, validation, and testing.
            model (str): The name of the model to use.
            tmp_dir_path (Path): The path to the temporary directory to use for storing checkpoints and exported models.

        Steps:
        1. Setup Engine
        2. Build Model from model name
        3. Training (1 epochs)
        4. Validation & Testing
        5. Prediction with single image
        6. Export Openvino IR Model

        Returns:
            None
        """
        # Setup Engine
        engine = Engine(work_dir=tmp_dir_path)
        built_model = get_model(model=model)

        # Train (1 epochs)
        results = engine.train(
            model=built_model,
            train_dataloader=dataset.train_dataloader(batch_size=2),
            val_dataloader=dataset.val_dataloader(),
            max_epochs=1,
        )
        assert "model" in results
        assert "checkpoint" in  results
        assert isinstance(results["model"], torch.nn.Module)
        assert isinstance(results["checkpoint"], str)
        assert Path(results["checkpoint"]).exists()

        # Validation
        val_score = engine.validate(val_dataloader=dataset.val_dataloader())
        # TODO: Check Validation results

        # Test
        test_score = engine.test(test_dataloader=dataset.test_dataloader())
        # TODO: Check Test results

        # Prediction with single image
        pred_result = engine.predict(
            model=results["model"],
            checkpoint=results["checkpoint"],
            img=PREDICTION_SAMPLE,
        )
        assert isinstance(pred_result, list)
        assert len(pred_result) == 1
        assert "pred_boxes" in pred_result[0]
        assert len(pred_result[0]["pred_boxes"]) > 0

        # Export Openvino IR Model
        export_output = engine.export(
            checkpoint=results["checkpoint"],
        )
        assert isinstance(export_output, dict)
        assert "outputs" in export_output
        assert isinstance(export_output["outputs"], dict)
        assert "bin" in export_output["outputs"]
        assert "xml" in export_output["outputs"]
        assert Path(export_output["outputs"]["bin"]).exists()
        assert Path(export_output["outputs"]["xml"]).exists()
