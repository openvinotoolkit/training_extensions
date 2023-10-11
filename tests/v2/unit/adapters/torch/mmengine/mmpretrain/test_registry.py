# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from otx.v2.adapters.torch.mmengine.mmpretrain.registry import MMPretrainRegistry


class TestMMPretrainRegistry:
    def test_init(self) -> None:
        registry = MMPretrainRegistry()
        assert registry.name == "mmpretrain"

    def test_init_with_name(self) -> None:
        registry = MMPretrainRegistry(name="test_registry")
        assert registry.name == "test_registry"
