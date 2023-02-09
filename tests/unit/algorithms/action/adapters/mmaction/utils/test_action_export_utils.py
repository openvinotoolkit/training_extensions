"""Unit Test for otx.algorithms.action.adapters.mmaction.utils.config_utils."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Any, List

import numpy as np
import torch
from mmaction.models import Recognizer3D
from mmcv.runner import BaseModule
from mmcv.utils import Config
from torch import nn

from otx.algorithms.action.adapters.mmaction.models.detectors.fast_rcnn import (
    AVAFastRCNN,
)
from otx.algorithms.action.adapters.mmaction.utils.export_utils import (
    _convert_sync_batch_to_normal_batch,
    export_model,
    get_frame_inds,
    onnx2openvino,
    preprocess,
    pytorch2onnx,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class MockRecognizer3D(Recognizer3D, BaseModule):
    """Mock class for Recognizer3D."""

    def __init__(self) -> None:
        super(BaseModule, self).__init__()

    def forward_dummy(self, inputs: Any) -> str:
        return "Forward function is replaced!"

    @staticmethod
    def cpu() -> Recognizer3D:
        return MockRecognizer3D()


class MockAVAFastRCNN(AVAFastRCNN):
    """Mock class for AVAFastRCNN."""

    def __init__(self) -> None:
        super(BaseModule, self).__init__()

    def add_detector(self) -> None:
        pass

    def patch_pools(self) -> None:
        pass

    def forward_infer(self, inputs: Any, img_metas: Any) -> str:
        return "Forward function is replaced!"

    @staticmethod
    def cpu() -> AVAFastRCNN:
        return MockAVAFastRCNN()


def _mock_sync_batchnorm(inputs):
    """Mock function for _sync_batch_to_normal_batch function.

    It returns its inputs
    """

    return inputs


@e2e_pytest_unit
def test_convert_sync_batch_to_normal_batch() -> None:
    """Test _convert_sync_batch_to_normal_batch function.

    <Steps>
        1. Create sample module, which has some Conv3D, SyncBatchNorm, BatchNorm3d ops
        2. Run _convert_sync_batch_to_normal_batch function to sample module
        3. Check SyncBatchNorm is changed into BatchNorm3d
        4. Check the other ops don't affect by this function
    """

    sample_module = nn.Sequential(
        nn.Conv3d(100, 100, 3), nn.SyncBatchNorm(100), nn.Conv3d(100, 100, 3), nn.BatchNorm3d(100)
    )
    output_module = _convert_sync_batch_to_normal_batch(sample_module)
    assert isinstance(output_module[0], nn.Conv3d)
    assert isinstance(output_module[1], nn.BatchNorm3d)
    assert isinstance(output_module[2], nn.Conv3d)
    assert isinstance(output_module[3], nn.BatchNorm3d)


@e2e_pytest_unit
def test_preprocess(mocker) -> None:
    """Test preprocess function.

    <Steps>
        1. Run preprocess function
        2. Check output imgs has proper shape: [1, 3, 3, 256, 256]
        2. Check output meta contains proper information
    """

    mocker.patch(
        "otx.algorithms.action.adapters.mmaction.utils.export_utils.get_frame_inds", return_value=np.array([2, 4, 6])
    )
    imgs, meta = preprocess(clip_len=8, width=256, height=256)
    assert list(imgs.shape) == [1, 3, 3, 256, 256]
    assert torch.all(meta[0][0]["img_shape"] == torch.Tensor([256.0, 256.0]))
    assert meta[0][0]["ori_shape"] == (5, 10)
    assert np.all(meta[0][0]["scale_factor"] == np.array([256 / 10, 256 / 5, 256 / 10, 256 / 5]))


@e2e_pytest_unit
def test_get_frame_inds() -> None:
    """Test get_frame_inds function.

    <Steps>
        1. Run get_frame_inds function for long enough inputs
        2. Check output indices
        3. Run get_frame_inds function for short inputs
        4. Check output indices
    """

    inputs = np.ndarray([1, 3, 50, 256, 256])
    outputs = get_frame_inds(inputs, 8, 2)
    assert np.all(outputs == np.array([17, 19, 21, 23, 25, 27, 29, 31]))

    inputs = np.ndarray([1, 3, 10, 256, 256])
    outputs = get_frame_inds(inputs, 8, 2)
    assert np.all(outputs == np.array([0, 2, 4, 6, 8, 9, 9, 9]))


@e2e_pytest_unit
def test_pytorch2onnx(mocker) -> None:
    """Test pytorch2onnx function.

    It checks whether model's forward function is replaced with proper alternatives
    <Steps>
        1. Run pytorch2onnx function for action detection model
        2. Check model's forward function is replaced with forward_infer function
        3. Run pytorch2onnx function for action classification model
        4. Check model's forward function is replaced with forward_dummy function
    """

    sample_tensor = torch.randn(1, 3, 32, 256, 256)
    sample_meta = [[{"meta": None}]]

    mocker.patch(
        "otx.algorithms.action.adapters.mmaction.utils.export_utils._convert_sync_batch_to_normal_batch",
        side_effect=_mock_sync_batchnorm,
    )
    mocker.patch(
        "otx.algorithms.action.adapters.mmaction.utils.export_utils.preprocess",
        return_value=(sample_tensor, sample_meta),
    )
    mocker.patch("otx.algorithms.action.adapters.mmaction.utils.export_utils.torch.onnx.export", return_value=True)

    model = MockAVAFastRCNN()
    pytorch2onnx(model, [1, 3, 32, 256, 256])
    assert model(sample_tensor) == "Forward function is replaced!"

    model = MockRecognizer3D()
    pytorch2onnx(model, [1, 1, 3, 8, 224, 224])
    assert model(sample_tensor) == "Forward function is replaced!"


@e2e_pytest_unit
def test_onnx2openvino(mocker) -> None:
    """Test onnx2openvino function.

    It checks the final openvino running command
    <Steps>
        1. Run onnx2openvino function for MockCfg
        2. Check openvino model conversion command
    """

    MockCfg = Config(
        {
            "model": {"pretrained": "./weights.pth"},
            "data": {
                "test": {
                    "test_mode": False,
                    "pipeline": [
                        {"type": "Empty"},
                        {"type": "Normalize", "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0], "to_bgr": True},
                    ],
                }
            },
        }
    )
    # TODO Check MockCfg

    MockModel = Config({"graph": {"output": [{"name": "1"}, {"name": "2"}, {"name": "3"}], "node": [{"output": "4"}]}})

    def _MockRun(command_line: List[str], **kwargs) -> None:
        """Mock function for subprocess.run function.

        It saves input command line into MockCfg so that we can check the command line
        """

        MockCfg.command_line = command_line

    mocker.patch("otx.algorithms.action.adapters.mmaction.utils.export_utils.onnx.load", return_value=MockModel)
    mocker.patch("otx.algorithms.action.adapters.mmaction.utils.export_utils.onnx.save", return_value=True)
    mocker.patch("otx.algorithms.action.adapters.mmaction.utils.export_utils.run", side_effect=_MockRun)

    onnx2openvino(
        cfg=MockCfg,
        onnx_model_path="temp.onnx",
        output_dir_path="./temp_dir/",
        layout="layout_sample",
        input_shape=[1, 3, 8, 224, 224],
        pruning_transformation=True,
    )

    assert MockCfg.model.pretrained is None
    assert MockCfg.data.test.test_mode is True
    assert MockCfg.command_line[0] in ["mo", "mo.py"]
    assert MockCfg.command_line[1].split("=")[-1] == "temp.onnx"
    assert MockCfg.command_line[2].split("=")[-1] == "[0.0, 0.0, 0.0]"
    assert MockCfg.command_line[3].split("=")[-1] == "[1.0, 1.0, 1.0]"
    assert MockCfg.command_line[4].split("=")[-1] == "./temp_dir/"
    assert MockCfg.command_line[7].split("=")[-1] == "layout_sample"
    assert MockCfg.command_line[8].split("=")[-1] == "[1, 3, 8, 224, 224]"
    assert MockCfg.command_line[9].split("--")[-1] == "reverse_input_channels"
    assert MockCfg.command_line[-1] == "Pruning"


@e2e_pytest_unit
def test_export_model(mocker) -> None:
    """Test export_model function.

    <Steps>
        1. Run export_model function for action classification model
        2. Check input_shape, and layout using MockCfg
        3. Run export_model funciton for action detection model
        4. Check input_shape, and layour using MockCfg
    """

    MockCfg = Config()

    def _MockPytorch2ONNX(*args, input_shape: List[int], **kwargs) -> None:
        """Mock function for pytorch2onnx.

        It saves input_shape to MockConfig
        """

        MockCfg.input_shape = input_shape

    def _MockONNX2OpenVINO(config, onnx_model_path, output_dir_path, layout, input_shape, **kwargs) -> None:
        """Mock function for onnx2pytorch.

        It saves layout to MockCfg
        """

        MockCfg.layout = layout

    mocker.patch(
        "otx.algorithms.action.adapters.mmaction.utils.export_utils.pytorch2onnx", side_effect=_MockPytorch2ONNX
    )
    mocker.patch(
        "otx.algorithms.action.adapters.mmaction.utils.export_utils.onnx2openvino", side_effect=_MockONNX2OpenVINO
    )

    export_model(MockRecognizer3D(), MockCfg)
    assert MockCfg.input_shape == [1, 1, 3, 8, 224, 224]
    assert MockCfg.layout == "??c???"

    export_model(MockAVAFastRCNN(), MockCfg)
    assert MockCfg.input_shape == [1, 3, 32, 256, 256]
    assert MockCfg.layout == "bctwh"
