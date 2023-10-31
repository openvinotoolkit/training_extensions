# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path

import pytest
import torch
from otx.v2.adapters.torch.lightning.visual_prompt import Dataset, Engine, get_model, list_models

from tests.v2.integration.test_helper import TASK_CONFIGURATION

# Test-related datasets are managed by tests.v2.integration.api.test_helper.
PREDICTION_SAMPLE = TASK_CONFIGURATION["visual_prompting"]["sample"]

MODELS: list = list_models("otx*")

class TestVisaulPromptAPI:
    """
    This class contains integration tests for the training API of the VisaulPrompt.
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
                train_data_roots=TASK_CONFIGURATION["visual_prompting"]["train_data_roots"],
                val_data_roots=TASK_CONFIGURATION["visual_prompting"]["val_data_roots"],
                test_data_roots=TASK_CONFIGURATION["visual_prompting"]["test_data_roots"],
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
            train_dataloader=dataset.train_dataloader(batch_size=6),
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
        assert isinstance(val_score, list)
        assert len(val_score) > 0

        # Test
        test_score = engine.test(test_dataloader=dataset.test_dataloader())
        assert isinstance(test_score, list)
        assert len(val_score) > 0

        # Prediction with single image
        pred_result = engine.predict(
            model=built_model,
            checkpoint=results["checkpoint"],
            img=PREDICTION_SAMPLE,
        )
        assert isinstance(pred_result, list)
        assert len(pred_result) == 1

        # Export Openvino IR Model
        export_output = engine.export(
            checkpoint=results["checkpoint"],
        )
        assert isinstance(export_output, dict)
        assert "outputs" in export_output
        assert isinstance(export_output["outputs"], dict)
        assert "onnx" in export_output["outputs"]
        assert "encoder" in export_output["outputs"]["onnx"]
        assert "sam" in export_output["outputs"]["onnx"]
        assert Path(export_output["outputs"]["onnx"]["encoder"]).exists()
        assert Path(export_output["outputs"]["onnx"]["sam"]).exists()

        assert "openvino" in export_output["outputs"]
        assert "encoder" in export_output["outputs"]["openvino"]
        assert "bin" in export_output["outputs"]["openvino"]["encoder"]
        assert "xml" in export_output["outputs"]["openvino"]["encoder"]
        assert Path(export_output["outputs"]["openvino"]["encoder"]["bin"]).exists()
        assert Path(export_output["outputs"]["openvino"]["encoder"]["xml"]).exists()
        assert "sam" in export_output["outputs"]["openvino"]
        assert "bin" in export_output["outputs"]["openvino"]["sam"]
        assert "xml" in export_output["outputs"]["openvino"]["sam"]
        assert Path(export_output["outputs"]["openvino"]["sam"]["bin"]).exists()
        assert Path(export_output["outputs"]["openvino"]["sam"]["xml"]).exists()
