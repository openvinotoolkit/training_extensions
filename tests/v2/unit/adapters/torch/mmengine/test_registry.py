# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from otx.v2.adapters.torch.mmengine.registry import MMEngineRegistry


class TestMMEngineRegistry:
    def test_init(self) -> None:
        registry = MMEngineRegistry()
        assert registry.name == "mmengine"

    def test_init_with_name(self) -> None:
        registry = MMEngineRegistry(name="test_registry")
        assert registry.name == "test_registry"
