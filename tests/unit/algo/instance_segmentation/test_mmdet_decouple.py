# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from otx.core.model.utils.mmdet import create_model
from otx.core.types.task import OTXTaskType
from otx.engine import Engine
from otx.engine.utils.auto_configurator import DEFAULT_CONFIG_PER_TASK


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

        train_metric = engine.train(max_epochs=1)
        assert len(train_metric) > 0

        test_metric = engine.test()
        assert len(test_metric) > 0

        predict_result = engine.predict()
        assert len(predict_result) > 0

        # Export IR Model
        exported_model_path: Path | dict[str, Path] = engine.export()
        if isinstance(exported_model_path, Path):
            assert exported_model_path.exists()
        test_metric_from_ov_model = engine.test(checkpoint=exported_model_path, accelerator="cpu")
        assert len(test_metric_from_ov_model) > 0
