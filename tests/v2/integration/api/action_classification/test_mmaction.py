"""Integration.API test for the action classfication."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch
from otx.v2.adapters.torch.mmengine.mmaction import Dataset, Engine, get_model, list_models

from tests.v2.integration.api.test_helper import assert_torch_dataset_api_is_working
from tests.v2.integration.test_helper import TASK_CONFIGURATION

from mmaction.utils import register_all_modules


def test_model_api() -> None:
    """
    Test the Model API by listing and getting models.

    This function tests the Model API by performing the following steps:
    1. List all available models and assert that the list is not empty.
    2. List all available models with the prefix "otx" and assert that the list is not empty.
    3. Get the first model from the list of all available models and assert that it is an instance of torch.nn.Module.
    4. Get the first model from the list of models with the prefix "otx" and assert that it is an instance of torch.nn.Module.
    """
    models = list_models()
    assert len(models) > 0
    otx_models = list_models("otx*")
    assert len(otx_models) > 0

    build_model = get_model(models[0])
    assert isinstance(build_model, torch.nn.Module)
    otx_model = get_model(otx_models[0])
    assert isinstance(otx_model, torch.nn.Module)


MODELS: list[str] = list_models("otx*")


class TestMMActionAPI:
    """
    This class contains integration tests for the training API of the MMPretrain library.
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
            train_data_roots=TASK_CONFIGURATION["action_classification"]["train_data_roots"],
            val_data_roots=TASK_CONFIGURATION["action_classification"]["val_data_roots"],
            test_data_roots=TASK_CONFIGURATION["action_classification"]["test_data_roots"],
        )

    def test_dataset_api(self, dataset: Dataset) -> None:
        """
        Test the Torch dataset & dataloader API for the given dataset.

        Args:
            dataset (Dataset): The dataset to test.

        Returns:
            None
        """
        register_all_modules(init_default_scope=True)
        assert_torch_dataset_api_is_working(dataset=dataset, train_data_size=2, val_data_size=2, test_data_size=2)

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
        built_model = get_model(model=model, num_classes=dataset.num_classes)
        
        # Train (1 epoch)
        results = engine.train(
            model=built_model,
            train_dataloader=dataset.train_dataloader(batch_size=10, num_workers=0),
            val_dataloader=dataset.val_dataloader(batch_size=1, num_workers=0),
            max_epochs=1,
        )
        assert "model" in results
        assert "checkpoint" in  results
        assert isinstance(results["model"], torch.nn.Module)
        assert isinstance(results["checkpoint"], str)
        assert Path(results["checkpoint"]).exists()

        # Validation
        val_score = engine.validate()
        assert "acc/top1" in val_score
        assert val_score["acc/top1"] > 0.0

        # Test
        test_score = engine.test(test_dataloader=dataset.test_dataloader())
        assert "acc/top1" in test_score
        assert test_score["acc/top1"] > 0.0

        # Prediction with images
        pred_result = engine.predict(
            model=results["model"],
            checkpoint=results["checkpoint"],
            img=TASK_CONFIGURATION["action_classification"]["sample"],
        )
        assert isinstance(pred_result, dict)
        assert len(pred_result['predictions']) == 1
        assert "rec_labels" in pred_result['predictions'][0]
        assert "rec_scores" in pred_result['predictions'][0]

        # Export Openvino IR Model
        export_output = engine.export(
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
