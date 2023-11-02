# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
from otx.v2.adapters.torch.modules.models.base_model import BaseOTXModel


class TestBaseOTXModel:
    def test_init(self) -> None:
        class InvalidMockModel(BaseOTXModel):
            """Test Model."""

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            InvalidMockModel()

        class MockModel(BaseOTXModel):
            """Test Model."""
            @property
            def callbacks(self) -> list:
                return []

            def export(
                self,
                export_dir: str | Path,
                export_type: str = "OPENVINO",
                precision: str | int | None = None
            ) -> dict:
                _, _, _ = export_dir, export_type, precision
                return {}

        model = MockModel()
        assert model.callbacks == []
        assert model.export(export_dir="test") == {}
