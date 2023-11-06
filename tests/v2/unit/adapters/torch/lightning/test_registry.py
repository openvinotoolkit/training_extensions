# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from otx.v2.adapters.torch.lightning.registry import LightningRegistry
from pytest_mock.plugin import MockerFixture


class TestLightningRegistry:
    def test_init(self) -> None:
        registry = LightningRegistry()
        assert registry.name == "lightning"

    def test_init_with_name(self) -> None:
        registry = LightningRegistry(name="test_registry")
        assert registry.name == "test_registry"

    def test_get(self, mocker: MockerFixture) -> None:
        registry = LightningRegistry()

        registry.register_module(name="test1", module=mocker.Mock)
        assert registry.get("test1") is not None
        assert registry.get("test2") is None
