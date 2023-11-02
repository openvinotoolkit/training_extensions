# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import torch
from otx.v2.adapters.torch.lightning import Dataset, get_model, list_models

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
    otx_models = list_models("otx_sam*")
    assert len(otx_models) > 0

    build_model = get_model(otx_models[0])
    assert isinstance(build_model, torch.nn.Module)
    otx_model = get_model(otx_models[0])
    assert isinstance(otx_model, torch.nn.Module)


def test_dataset_api() -> None:
    """
    Test the Torch dataset & dataloader API for the given dataset.

    Args:
        dataset (Dataset): The dataset to test.

    Returns:
        None
    """
    dataset = Dataset(
        task="visual_prompting",
        train_data_roots=TASK_CONFIGURATION["visual_prompting"]["train_data_roots"],
        val_data_roots=TASK_CONFIGURATION["visual_prompting"]["val_data_roots"],
        test_data_roots=TASK_CONFIGURATION["visual_prompting"]["test_data_roots"],
    )
    assert_torch_dataset_api_is_working(dataset=dataset, train_data_size=6, val_data_size=2, test_data_size=2)
