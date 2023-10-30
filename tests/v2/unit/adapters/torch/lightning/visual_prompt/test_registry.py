# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from otx.v2.adapters.torch.lightning.visual_prompt.registry import VisualPromptRegistry
from pytest_mock.plugin import MockerFixture


class TestVisualPromptRegistry:
    def test_init(self) -> None:
        registry = VisualPromptRegistry()
        assert registry.name == "visual_prompt"

    def test_init_with_name(self) -> None:
        registry = VisualPromptRegistry(name="test_registry")
        assert registry.name == "test_registry"

    def test_get(self, mocker: MockerFixture) -> None:
        registry = VisualPromptRegistry()

        registry.register_module(name="test1", module=mocker.Mock)
        assert registry.get("test1") is not None
        assert registry.get("test2") is None
