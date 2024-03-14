# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from otx.engine import Engine


class TestEngine:
    def test_from_config_with_model_name(self, tmp_path) -> None:
        model_name = "efficientnet_b0_light"
        data_root = "tests/assets/classification_dataset"
        task_type = "MULTI_CLASS_CLS"

        overriding = {
            "data.config.train_subset.batch_size": 2,
            "data.config.test_subset.subset_name": "TESTING",
        }

        engine = Engine.from_config(
            config=model_name,
            data_root=data_root,
            task=task_type,
            work_dir=tmp_path,
            **overriding,
        )

        assert engine is not None
        assert engine.datamodule.config.train_subset.batch_size == 2
        assert engine.datamodule.config.test_subset.subset_name == "TESTING"
