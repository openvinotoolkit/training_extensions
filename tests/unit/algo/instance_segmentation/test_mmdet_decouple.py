# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from pathlib import Path

from otx.core.types.task import OTXTaskType
from otx.engine import Engine
from otx.engine.utils.auto_configurator import DEFAULT_CONFIG_PER_TASK

from otx.core.utils.build import build_mm_model, get_classification_layers


if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch import nn


def create_model(config: DictConfig, load_from: str | None) -> tuple[nn.Module, dict[str, dict[str, int]]]:
    """Create a model from mmdet Model registry.

    Args:
        config (DictConfig): Model configuration.
        load_from (str | None): Model weight file path.

    Returns:
        tuple[nn.Module, dict[str, dict[str, int]]]: Model instance and classification layers.
    """
    from mmengine.registry import MODELS
    classification_layers = get_classification_layers(config, MODELS, "model.")
    return build_mm_model(config, MODELS, load_from), classification_layers


class TestDecoupleMMDetInstanceSeg:
    def test_maskrcnn(self, tmp_path: Path) -> None:

        tmp_path_train = tmp_path / OTXTaskType.INSTANCE_SEGMENTATION
        engine = Engine.from_config(
            config_path=DEFAULT_CONFIG_PER_TASK[OTXTaskType.INSTANCE_SEGMENTATION],
            data_root="tests/assets/car_tree_bug",
            work_dir=tmp_path_train,
            device="gpu",
        )

        new_model, classification_layers = create_model(engine.model.config, engine.model.load_from)
        engine.model.model = new_model

        train_metric = engine.train(max_epochs=2)
        assert len(train_metric) > 0

        test_metric = engine.test()
        assert len(test_metric) > 0

        predict_result = engine.predict()
        assert len(predict_result) > 0

        # Export IR Model
        # exported_model_path: Path | dict[str, Path] = engine.export()
        # if isinstance(exported_model_path, Path):
        #     assert exported_model_path.exists()
        # elif isinstance(exported_model_path, dict):
        #     for key, value in exported_model_path.items():
        #         assert value.exists(), f"{value} for {key} doesn't exist."
        # else:
        #     AssertionError(f"Exported model path is not a Path or a dictionary of Paths: {exported_model_path}")

        # Test with IR Model
        # test_metric_from_ov_model = engine.test(checkpoint=exported_model_path, accelerator="cpu")
        # assert len(test_metric_from_ov_model) > 0
