# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools

import pytest
from datumaro import Dataset as DmDataset
from otx.core.config.data import SamplerConfig, SubsetConfig
from otx.core.data.factory import OTXDatasetFactory
from otx.core.data.mem_cache import MemCacheHandlerSingleton
from otx.core.types.task import OTXTaskType
from otx.engine.utils.auto_configurator import AutoConfigurator


def _test_augmentation(
    recipe: str,
    target_dataset_per_task: dict,
    configurable_augs: list[str],
) -> None:
    # Load recipe
    recipe_tokens = recipe.split("/")
    model_name = recipe_tokens[-1].split(".")[0]
    task_name = recipe_tokens[-2]
    task = OTXTaskType(task_name.upper())
    config = AutoConfigurator(
        data_root=target_dataset_per_task[task_name],
        task=task,
        model_name=model_name,
    ).config
    train_config = config["data"]["train_subset"]
    train_config["input_size"] = (32, 32)

    # Load dataset
    dm_dataset = DmDataset.import_from(
        target_dataset_per_task[task_name],
        format=config["data"]["data_format"],
    )
    mem_cache_handler = MemCacheHandlerSingleton.create(
        mode="sinlgeprocessing",
        mem_size=0,
    )

    # Evaluate all on/off aug combinations
    img_shape = None
    for switches in itertools.product([True, False], repeat=len(configurable_augs)):
        # Configure on/off
        for aug_name, switch in zip(configurable_augs, switches):
            aug_found = False
            for aug_config in train_config["transforms"]:
                if aug_name in aug_config["class_path"]:
                    aug_config["enable"] = switch
                    aug_found = True
                    break
            assert aug_found, f"{aug_name} not found in {recipe}"
        # Create dataset
        dataset = OTXDatasetFactory.create(
            task=task,
            dm_subset=dm_dataset,
            cfg_subset=SubsetConfig(sampler=SamplerConfig(**train_config.pop("sampler", {})), **train_config),
            mem_cache_handler=mem_cache_handler,
        )

        # Check if all aug combinations are size-compatible
        data = dataset[0]
        if not img_shape:
            img_shape = data.img_info.img_shape
        else:
            assert img_shape == data.img_info.img_shape


CLS_RECIPES = [
    recipe for recipe in pytest.RECIPE_LIST if "_cls" in recipe and "semi" not in recipe and "tv_" not in recipe
]
DET_RECIPES = [recipe for recipe in pytest.RECIPE_LIST if "/detection/" in recipe]
INST_SEG_RECIPES = [recipe for recipe in pytest.RECIPE_LIST if "/instance_segmentation/" in recipe]


@pytest.mark.parametrize("recipe", CLS_RECIPES + DET_RECIPES + INST_SEG_RECIPES)
def test_augmentation(
    recipe: str,
    fxt_target_dataset_per_task: dict,
):
    configurable_augs = [
        "PhotoMetricDistortion",
        "RandomAffine",
        "RandomVerticalFlip",
        "GaussianBlur",
        "GaussianNoise",
    ]
    _test_augmentation(recipe, fxt_target_dataset_per_task, configurable_augs)
