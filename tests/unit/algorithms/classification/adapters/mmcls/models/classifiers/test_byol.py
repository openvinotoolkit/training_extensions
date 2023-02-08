from copy import deepcopy
from typing import Any, Dict

import pytest
import torch
import torch.nn as nn

from otx.algorithms.classification.adapters.mmcls import BYOL
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
class TestBYOL:
    """Test BYOL."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch, mocker) -> None:
        class MockBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.pipeline = nn.Sequential(nn.Conv2d(3, 1, (1, 1), bias=False), nn.Conv2d(1, 1, (1, 1), bias=False))

            def init_weights(self, pretrained=None):
                pass

            def forward(self, x):
                return self.pipeline(x)

        class MockNeck(nn.Sequential):
            def __init__(self):
                super().__init__(nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False))

            def init_weights(self, init_linear=None):
                pass

        class MockHead(nn.Sequential):
            def __init__(self):
                super().__init__(nn.Linear(2, 2, bias=False))

            def init_weights(self, init_linear=None):
                pass

            def forward(self, *args, **kwargs):
                return {"loss": torch.Tensor(1)}

        def build_mock_backbone(*args, **kwargs):
            return MockBackbone()

        def build_mock_neck(*args, **kwargs):
            return MockNeck()

        def build_mock_head(*args, **kwargs):
            return MockHead()

        monkeypatch.setattr(
            "otx.algorithms.classification.adapters.mmcls.models.classifiers.byol.build_backbone", build_mock_backbone
        )
        monkeypatch.setattr(
            "otx.algorithms.classification.adapters.mmcls.models.classifiers.byol.build_neck", build_mock_neck
        )
        monkeypatch.setattr(
            "otx.algorithms.classification.adapters.mmcls.models.classifiers.byol.build_head", build_mock_head
        )

        self.byol = BYOL(backbone={}, neck={}, head={})

    @e2e_pytest_unit
    def test_init_weights(self) -> None:
        """Test init_weights function."""
        for param_ol, param_tgt in zip(self.byol.online_backbone.parameters(), self.byol.target_backbone.parameters()):
            assert torch.all(param_ol == param_tgt)
            assert param_ol.requires_grad
            assert not param_tgt.requires_grad

        for param_ol, param_tgt in zip(
            self.byol.online_projector.parameters(), self.byol.target_projector.parameters()
        ):
            assert torch.all(param_ol == param_tgt)
            assert param_ol.requires_grad
            assert not param_tgt.requires_grad

    @e2e_pytest_unit
    def test_momentum_update(self) -> None:
        """Test _momentum_update function."""
        original_params = {"backbone": [], "projector": []}
        for param_tgt in self.byol.target_backbone.parameters():
            param_tgt.data *= 2.0
            original_params["backbone"].append(deepcopy(param_tgt))

        for param_tgt in self.byol.target_projector.parameters():
            param_tgt.data *= 2.0
            original_params["projector"].append(deepcopy(param_tgt))

        self.byol.momentum_update()

        for param_ol, param_tgt, orig_tgt in zip(
            self.byol.online_backbone.parameters(), self.byol.target_backbone.parameters(), original_params["backbone"]
        ):
            assert torch.all(
                param_tgt.data == orig_tgt * self.byol.momentum + param_ol.data * (1.0 - self.byol.momentum)
            )

        for param_ol, param_tgt, orig_tgt in zip(
            self.byol.online_projector.parameters(),
            self.byol.target_projector.parameters(),
            original_params["projector"],
        ):
            assert torch.all(
                param_tgt.data == orig_tgt * self.byol.momentum + param_ol.data * (1.0 - self.byol.momentum)
            )

    @e2e_pytest_unit
    def test_train_step(self) -> None:
        """Test train_step function wraps forward and _parse_losses."""
        img1 = torch.randn((1, 3, 2, 2))
        img2 = torch.randn((1, 3, 2, 2))

        outputs = self.byol.train_step(data=dict(img1=img1, img2=img2), optimizer=None)

        assert "loss" in outputs
        assert "log_vars" in outputs
        assert "num_samples" in outputs

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "orig_state_dict,prefix,expected",
        [
            ({"online_backbone.layer": 1}, "", {"layer": 1}),
            ({"backbone.layer": 1}, "", {"backbone.layer": 1}),
            ({"backbone.layer": 1}, "backbone.", {"backbone.layer": 1}),
        ],
    )
    def test_state_dict_hook(self, orig_state_dict: Dict[str, Any], prefix: str, expected: Dict[str, Any]) -> None:
        """Test state_dict_hook function."""
        new_state_dict = BYOL.state_dict_hook(module=self.byol, state_dict=orig_state_dict, prefix=prefix)

        assert new_state_dict == expected
