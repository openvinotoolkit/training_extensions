# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from otx.tools.converter import ConfigConverter


class TestConfigConverter:
    def test_convert(self):
        config = ConfigConverter.convert("tests/assets/geti-configs/det.json")

        assert config["data"]["config"]["mem_cache_size"] == "100MB"
        assert config["data"]["config"]["train_subset"]["batch_size"] == 16
        assert config["data"]["config"]["val_subset"]["batch_size"] == 8
        assert config["data"]["config"]["test_subset"]["batch_size"] == 8
        assert config["model"]["init_args"]["optimizer"]["init_args"]["lr"] == 0.01
        assert config["model"]["init_args"]["scheduler"]["init_args"]["num_warmup_steps"] == 6
        assert config["max_epochs"] == 50
        assert config["data"]["config"]["train_subset"]["num_workers"] == 8
        assert config["data"]["config"]["val_subset"]["num_workers"] == 8
        assert config["data"]["config"]["test_subset"]["num_workers"] == 8
        assert config["callbacks"][0]["init_args"]["patience"] == 4
        assert config["data"]["config"]["tile_config"]["enable_tiler"] is True
        assert config["data"]["config"]["tile_config"]["overlap"] == 0.5

    def test_instantiate(self, tmp_path):
        data_root = "tests/assets/car_tree_bug"
        config = ConfigConverter.convert(config_path="tests/assets/geti-configs/det.json")
        engine, train_kwargs = ConfigConverter.instantiate(
            config=config,
            work_dir=tmp_path,
            data_root=data_root,
        )
        assert engine.work_dir == tmp_path

        assert engine.datamodule.config.data_root == data_root
        assert engine.datamodule.config.mem_cache_size == "100MB"
        assert engine.datamodule.config.train_subset.batch_size == 16
        assert engine.datamodule.config.val_subset.batch_size == 8
        assert engine.datamodule.config.test_subset.batch_size == 8
        assert engine.datamodule.config.train_subset.num_workers == 8
        assert engine.datamodule.config.val_subset.num_workers == 8
        assert engine.datamodule.config.test_subset.num_workers == 8
        assert engine.datamodule.config.tile_config.enable_tiler

        assert len(train_kwargs["callbacks"]) == len(config["callbacks"])
        assert train_kwargs["callbacks"][0].patience == 4
        assert len(train_kwargs["logger"]) == len(config["logger"])
        assert train_kwargs["max_epochs"] == 50
