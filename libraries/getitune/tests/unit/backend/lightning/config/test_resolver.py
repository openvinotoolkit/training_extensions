# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from omegaconf import OmegaConf


class TestResolver:
    def test_as_int_tuple(self) -> None:
        cfg_str = """
        mem_cache_img_max_size: ${as_int_tuple:1333,800}
        """
        cfg = OmegaConf.create(cfg_str)
        assert isinstance(cfg.mem_cache_img_max_size, tuple)
        assert cfg.mem_cache_img_max_size == (1333, 800)

    def test_as_torch_dtype(self) -> None:
        cfg_str = """
        uint8: ${as_torch_dtype:torch.uint8}
        int64: ${as_torch_dtype:torch.int64}
        float32: ${as_torch_dtype:torch.float32}
        """
        cfg = OmegaConf.create(cfg_str)

        assert cfg.uint8 == torch.uint8
        assert cfg.int64 == torch.int64
        assert cfg.float32 == torch.float32
