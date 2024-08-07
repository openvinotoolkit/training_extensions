# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from otx.tools.converter import ConfigConverter


class TestConfigConverter:
    def test_convert(self):
        config = ConfigConverter.convert("tests/assets/geti-configs/det.json")

        assert config["data"]["mem_cache_size"] == "100MB"
        assert config["data"]["train_subset"]["batch_size"] == 16
        assert config["data"]["val_subset"]["batch_size"] == 8
        assert config["data"]["test_subset"]["batch_size"] == 8
        assert config["model"]["init_args"]["optimizer"]["init_args"]["lr"] == 0.01
        assert config["model"]["init_args"]["scheduler"]["init_args"]["num_warmup_steps"] == 6
        assert config["max_epochs"] == 50
        assert config["data"]["train_subset"]["num_workers"] == 8
        assert config["data"]["val_subset"]["num_workers"] == 8
        assert config["data"]["test_subset"]["num_workers"] == 8
        assert config["callbacks"][0]["init_args"]["patience"] == 10
        assert config["data"]["tile_config"]["enable_tiler"] is True
        assert config["data"]["tile_config"]["overlap"] == 0.5

    def test_convert_task_overriding(self):
        default_config = ConfigConverter.convert("tests/assets/geti-configs/cls.json")
        assert default_config["engine"]["task"] == "MULTI_CLASS_CLS"

        override_config = ConfigConverter.convert("tests/assets/geti-configs/cls.json", task="MULTI_LABEL_CLS")
        assert override_config["engine"]["task"] == "MULTI_LABEL_CLS"

        override_config = ConfigConverter.convert("tests/assets/geti-configs/cls.json", task="H_LABEL_CLS")
        assert override_config["engine"]["task"] == "H_LABEL_CLS"

        with pytest.raises(SystemExit):
            ConfigConverter.convert("tests/assets/geti-configs/cls.json", task="DETECTION")

    def test_instantiate(self, tmp_path):
        data_root = "tests/assets/car_tree_bug"
        config = ConfigConverter.convert(config_path="tests/assets/geti-configs/det.json")
        engine, train_kwargs = ConfigConverter.instantiate(
            config=config,
            work_dir=tmp_path,
            data_root=data_root,
        )
        assert engine.work_dir == tmp_path

        assert engine.datamodule.data_root == data_root
        assert engine.datamodule.mem_cache_size == "100MB"
        assert engine.datamodule.train_subset.batch_size == 16
        assert engine.datamodule.val_subset.batch_size == 8
        assert engine.datamodule.test_subset.batch_size == 8
        assert engine.datamodule.train_subset.num_workers == 8
        assert engine.datamodule.val_subset.num_workers == 8
        assert engine.datamodule.test_subset.num_workers == 8
        assert engine.datamodule.tile_config.enable_tiler

        assert len(train_kwargs["callbacks"]) == len(config["callbacks"])
        assert train_kwargs["callbacks"][0].patience == 10
        assert len(train_kwargs["logger"]) == len(config["logger"])
        assert train_kwargs["max_epochs"] == 50
