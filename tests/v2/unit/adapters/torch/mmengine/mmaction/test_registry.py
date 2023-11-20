"""Unit-test for the registry API for MMAction."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from otx.v2.adapters.torch.mmengine.mmaction.registry import MMActionRegistry


class TestMMActionRegistry:
    def test_init(self) -> None:
        registry = MMActionRegistry()
        assert registry.name == "mmaction"

    def test_init_with_name(self) -> None:
        registry = MMActionRegistry(name="test_registry")
        assert registry.name == "test_registry"
