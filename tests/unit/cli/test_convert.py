# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from otx.cli.convert import BaseConverter


def test_convert():
    config = BaseConverter().convert("tests/assets/geti-configs/det.json")
    assert config["data"]["config"]["mem_cache_size"] == "0.1GB"
    assert config["data"]["config"]["train_subset"]["batch_size"] == 16
    assert config["data"]["config"]["val_subset"]["batch_size"] == 8
    assert config["data"]["config"]["test_subset"]["batch_size"] == 8
    assert config["optimizer"]["init_args"]["lr"] == 0.01
    assert config["scheduler"][0]["init_args"]["num_warmup_steps"] == 6
    assert config["max_epoch"] == 50
    assert config["data"]["config"]["train_subset"]["num_workers"] == 8
    assert config["data"]["config"]["val_subset"]["num_workers"] == 8
    assert config["data"]["config"]["test_subset"]["num_workers"] == 8
    assert config["callbacks"][0]["init_args"]["patience"] == 4
    assert config["data"]["config"]["tile_config"]["enable_tiler"] is True
    assert config["data"]["config"]["tile_config"]["overlap"] == 0.5
