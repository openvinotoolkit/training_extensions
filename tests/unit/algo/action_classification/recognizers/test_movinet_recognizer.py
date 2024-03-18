# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of movinet recognizer."""


import pytest
import torch
from otx.algo.action_classification.recognizers.movinet_recognizer import MoViNetRecognizer


class TestMoViNetRecognizer:
    @pytest.fixture()
    def fxt_movinet_recognizer(self, mocker) -> MoViNetRecognizer:
        mocker.patch.object(MoViNetRecognizer, "__init__", return_value=None)
        return MoViNetRecognizer()

    def test_state_dict_hook(self, fxt_movinet_recognizer: MoViNetRecognizer) -> None:
        state_dict = {"cls_head.conv1": torch.Tensor([0]), "backbone.conv2": torch.Tensor([1])}
        fxt_movinet_recognizer.state_dict_hook(torch.nn.Module(), state_dict)
        assert state_dict["conv1"] == torch.Tensor([0])
        assert state_dict["conv2"] == torch.Tensor([1])

    def test_load_stete_dict_pre_hook(self, fxt_movinet_recognizer: MoViNetRecognizer) -> None:
        state_dict = {"classifier.conv1": torch.Tensor([0]), "model.conv2": torch.Tensor([1])}
        fxt_movinet_recognizer.load_state_dict_pre_hook(torch.nn.Module(), state_dict, "model.")
        assert state_dict["cls_head.classifier.conv1"] == torch.Tensor([0])
        assert state_dict["model.backbone.conv2"] == torch.Tensor([1])
