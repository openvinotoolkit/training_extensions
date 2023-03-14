"""Unit Test for otx.algorithms.action.adapters.mmaction.models.recognizers.movinet_recognizer."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from copy import deepcopy

import pytest
import torch
from mmaction.models.recognizers.recognizer3d import Recognizer3D

from otx.algorithms.action.adapters.mmaction.models.recognizers.movinet_recognizer import (
    MoViNetRecognizer,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class MockOTXMoViNet:
    pass


class MockModule:
    def __init__(self):
        self.backbone = MockOTXMoViNet()
        self._state_dict = {
            "classifier.0.conv_1.conv3d.weight": torch.rand(1, 1),
            "conv1.conv_1.conv3d.weight": torch.rand(1, 1),
        }
        self.is_export = False

    def state_dict(self):
        return self._state_dict


class TestMoViNetRecognizer:
    @pytest.fixture(autouse=True)
    def setup(self, mocker) -> None:
        mocker.patch.object(Recognizer3D, "__init__", return_value=None)
        MoViNetRecognizer._register_state_dict_hook = mocker.MagicMock()
        MoViNetRecognizer._register_load_state_dict_pre_hook = mocker.MagicMock()
        self.recognizer = MoViNetRecognizer()
        self.prefix = ""

    @e2e_pytest_unit
    def test_load_state_dict_pre_hook(self) -> None:
        """Test load_state_dict_pre_hook function."""
        module = MockModule()
        state_dict = module.state_dict()
        self.recognizer.load_state_dict_pre_hook(module, state_dict, prefix=self.prefix)

        for key in state_dict:
            if "classifier" in key:
                assert "cls_head.classifier.0.conv_1.conv3d.weight" in state_dict
            else:
                assert "backbone.conv1.conv_1.conv3d.weight" in state_dict

    @e2e_pytest_unit
    def test_state_dict_hook(self):
        """Test state_dict_hook function."""
        module = MockModule()
        state_dict = module.state_dict()
        state_dict_copy = deepcopy(state_dict)
        self.recognizer.load_state_dict_pre_hook(module, state_dict, prefix=self.prefix)
        # backward state dict
        self.recognizer.state_dict_hook(module, state_dict, prefix=self.prefix)

        assert state_dict.keys() == state_dict_copy.keys()
