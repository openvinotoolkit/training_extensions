# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch
from otx.v2.adapters.torch.mmengine.mmdet import Dataset, Engine, get_model, list_models

from tests.v2.integration.api.test_helper import assert_torch_dataset_api_is_working
from tests.v2.integration.test_helper import TASK_CONFIGURATION


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


MODELS: list = list_models("otx*")


class TestMMDetAPI:
    """
    This class contains integration tests for the training API of the MMDet library.
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
            train_data_roots=TASK_CONFIGURATION["instance_segmentation"]["train_data_roots"],
            val_data_roots=TASK_CONFIGURATION["instance_segmentation"]["val_data_roots"],
            test_data_roots=TASK_CONFIGURATION["instance_segmentation"]["test_data_roots"],
            task="instance_segmentation",
        )

    def test_dataset_api(self, dataset: Dataset) -> None:
        """
        Test the Torch dataset & dataloader API for the given dataset.

        Args:
            dataset (Dataset): The dataset to test.

        Returns:
            None
        """
        assert_torch_dataset_api_is_working(dataset=dataset, train_data_size=6, val_data_size=3, test_data_size=3)

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

        # Train (1 epochs)
        results = engine.train(
            model=built_model,
            train_dataloader=dataset.train_dataloader(),
            val_dataloader=dataset.val_dataloader(),
            max_epochs=1,
        )
        assert "model" in results
        assert "checkpoint" in results
        assert isinstance(results["model"], torch.nn.Module)
        assert isinstance(results["checkpoint"], str)
        assert Path(results["checkpoint"]).exists()

        # Validation
        val_score = engine.validate()
        assert "pascal_voc/mAP" in val_score
        assert val_score["pascal_voc/mAP"] >= 0.0

        # Test
        test_score = engine.test(test_dataloader=dataset.test_dataloader())
        assert "pascal_voc/mAP" in test_score
        assert test_score["pascal_voc/mAP"] >= 0.0

        # Prediction with single image
        pred_result = engine.predict(
            model=results["model"],
            checkpoint=results["checkpoint"],
            img=TASK_CONFIGURATION["detection"]["sample"],
        )
        assert isinstance(pred_result, list)
        assert len(pred_result) == 1

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
