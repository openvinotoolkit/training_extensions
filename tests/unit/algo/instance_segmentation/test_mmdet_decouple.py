# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from otx.core.types.task import OTXTaskType
from otx.core.utils.build import build_mm_model, get_classification_layers
from otx.engine import Engine
from otx.engine.utils.auto_configurator import DEFAULT_CONFIG_PER_TASK

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch import nn


def create_model(config: DictConfig, load_from: str | None) -> tuple[nn.Module, dict[str, dict[str, int]]]:
    """Create a model from mmengine Model registry.

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
            device="cpu",
        )

        new_model, _ = create_model(engine.model.config, engine.model.load_from)
        engine.model.model = new_model

        train_metric = engine.train(max_epochs=2)
        assert len(train_metric) > 0

        test_metric = engine.test()
        assert len(test_metric) > 0

        predict_result = engine.predict()
        assert len(predict_result) > 0

        # TODO(Eugene): add export IR test
        # https://github.com/openvinotoolkit/training_extensions/pull/3281
