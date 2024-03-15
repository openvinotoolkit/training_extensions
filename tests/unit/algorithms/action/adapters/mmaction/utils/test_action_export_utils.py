"""Unit Test for otx.algorithms.action.adapters.mmaction.utils.config_utils."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Any

import torch
from mmaction.models import Recognizer3D
from mmcv.runner import BaseModule
from mmcv.utils import Config
from torch import nn

from otx.algorithms.action.adapters.mmaction.models.detectors.fast_rcnn import (
    AVAFastRCNN,
)
from otx.algorithms.action.adapters.mmaction.utils.export_utils import (
    Exporter,
    _convert_sync_batch_to_normal_batch,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class MockRecognizer3D(Recognizer3D, BaseModule):
    """Mock class for Recognizer3D."""

    def __init__(self) -> None:
        super(BaseModule, self).__init__()

    def forward(self, inputs: Any) -> str:
        return "Forward function is replaced!"

    def load_state_dict(self, weights) -> Recognizer3D:
        pass


class MockAVAFastRCNN(AVAFastRCNN):
    """Mock class for AVAFastRCNN."""

    def __init__(self) -> None:
        super(BaseModule, self).__init__()
        self.deploy_cfg = None

    def patch_for_export(self) -> None:
        pass

    def forward_infer(self, inputs: Any, img_metas: Any) -> str:
        return "Forward function is replaced!"

    def load_state_dict(self, weights) -> AVAFastRCNN:
        pass


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


class MockTaskProcessor:
    """Mock class of task_processor."""

    def __init__(self, model_cfg, deploy_cfg, device):
        self.model_cfg = model_cfg

    def init_pytorch_model(self, weights):
        if self.model_cfg.model == "cls":
            return MockRecognizer3D()
        return MockAVAFastRCNN()


def mock_build_task_processor(model_cfg, deploy_cfg, device):
    return MockTaskProcessor(model_cfg, deploy_cfg, device)


class TestExporter:
    """Test class for Exporter."""

    @e2e_pytest_unit
    def test_init(self, mocker) -> None:
        """Test __init__ function.

        <Steps>
            1. Create mock task_processor
            2. Create mock Recognizer3D using task_processor
            3. Get inputs
            4. Create mock AVAFastRCNN using task_processor
            5. Get inputs
            6. Check mo options when half precision
        """

        mocker.patch(
            "otx.algorithms.action.adapters.mmaction.utils.export_utils.build_task_processor",
            side_effect=mock_build_task_processor,
        )

        recipe_cfg = Config(dict(model="cls"))
        deploy_cfg = Config(
            dict(
                backend_config=dict(
                    type="openvino",
                    mo_options={},
                    model_inputs=[dict(opt_shapes=dict(input=[1, 1, 3, 32, 224, 224]))],
                )
            )
        )
        exporter = Exporter(recipe_cfg, None, deploy_cfg, "./tmp_dir/openvino", False, False)
        assert isinstance(exporter.model, Recognizer3D)
        assert exporter.input_tensor.shape == torch.Size([1, 1, 3, 32, 224, 224])
        assert exporter.input_metas is None

        recipe_cfg = Config(dict(model="det"))
        deploy_cfg = Config(
            dict(
                backend_config=dict(
                    type="openvino",
                    mo_options={},
                    model_inputs=[dict(opt_shapes=dict(input=[1, 3, 32, 224, 224]))],
                )
            )
        )
        exporter = Exporter(recipe_cfg, None, deploy_cfg, "./tmp_dir/openvino", False, False)
        assert isinstance(exporter.model, AVAFastRCNN)
        assert exporter.input_tensor.shape == torch.Size([1, 3, 32, 224, 224])
        assert exporter.input_metas is not None

        exporter = Exporter(recipe_cfg, None, deploy_cfg, "./tmp_dir/openvino", True, False)
        assert exporter.deploy_cfg.backend_config.mo_options["flags"] == ["--compress_to_fp16"]

    @e2e_pytest_unit
    def test_export(self, mocker) -> None:
        """Test export function."""

        mocker.patch("otx.algorithms.action.adapters.mmaction.utils.export_utils.export", return_value=True)
        mocker.patch("otx.algorithms.action.adapters.mmaction.utils.export_utils.from_onnx", return_value=True)
        mocker.patch(
            "otx.algorithms.action.adapters.mmaction.utils.export_utils.build_task_processor",
            side_effect=mock_build_task_processor,
        )

        recipe_cfg = Config(dict(model="cls"))
        deploy_cfg = Config(
            dict(
                backend_config=dict(
                    type="openvino",
                    mo_options={},
                    model_inputs=[dict(opt_shapes=dict(input=[1, 1, 3, 32, 224, 224]))],
                ),
                ir_config=dict(input_names=["input"], output_names=["output"]),
            )
        )
        exporter = Exporter(recipe_cfg, None, deploy_cfg, "./tmp_dir/openvino", False, False)
        exporter.export()
