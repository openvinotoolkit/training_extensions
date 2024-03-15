"""Unit-Test Case for otx.cli.builder.builder."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from pathlib import Path

import mmcv
import pytest
from mmcv.utils import Registry
from torch import nn

from otx.algorithms.common.adapters.mmcv.utils.config_utils import OTXConfig
from otx.cli.builder.builder import (
    Builder,
    get_backbone_out_channels,
    update_backbone_args,
    update_channels,
)
from otx.cli.utils.importing import get_otx_root_path
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestOTXCLIBuilder:
    """Check Builder's function is working well.

    1. Check "Builder.build_backbone_config" function that generate backbone configuration file is working well
    <Steps>
        1. Generate backbone config file (mmcls.MMOVBackbone)
        2. Raise ValueError with wrong output_path

    2. Check "Builder.merge_backbone" function that update model config with new backbone is working well
    <Steps>
        1. Update model config with mmcls.ResNet backbone (default model.backbone: otx.OTXEfficientNet)
        2. Raise ValueError with wrong model_config_path
        3. Raise ValueError with wrong backbone_config_path
        4. Update model config without backbone's out_indices
        5. Update model config with backbone's pretrained path
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir_path: str) -> None:
        self.otx_builder = Builder()
        self.otx_root = get_otx_root_path()
        self.tmp_dir_path = tmp_dir_path if isinstance(tmp_dir_path, Path) else Path(tmp_dir_path)

    @e2e_pytest_unit
    @pytest.mark.parametrize("backbone_type", ["mmcls.MMOVBackbone"])
    def test_builder_build_backbone_config_generate_backbone(self, backbone_type: str) -> None:
        """Generate backbone config file (mmcls.MMOVBackbone)."""
        tmp_backbone_path = self.tmp_dir_path / "backbone.yaml"
        self.otx_builder.build_backbone_config(backbone_type, tmp_backbone_path)
        assert tmp_backbone_path.exists()
        backbone_config = mmcv.load(str(tmp_backbone_path))
        assert backbone_config["backbone"]["type"] == backbone_type

    @e2e_pytest_unit
    @pytest.mark.parametrize("backbone_type", ["mmcls.MMOVBackbone"])
    def test_builder_build_backbone_config_abnormal_output_path(self, backbone_type: str) -> None:
        """Raise ValueError with wrong output_path."""
        tmp_backbone_path = self.tmp_dir_path / "wrong.path"
        with pytest.raises(ValueError):
            self.otx_builder.build_backbone_config(backbone_type, tmp_backbone_path)

    @e2e_pytest_unit
    def test_builder_merge_backbone_abnormal_model_path(self) -> None:
        """Raise ValueError with wrong model_config_path."""
        workspace_path = self.tmp_dir_path / "test_builder_merge_backbone"
        tmp_backbone_path = workspace_path / "backbone.yaml"
        with pytest.raises(ValueError):
            self.otx_builder.merge_backbone("unexpected", tmp_backbone_path)

    @e2e_pytest_unit
    def test_builder_merge_backbone_abnormal_backbone_path(self) -> None:
        """Raise ValueError with wrong backbone_config_path."""
        workspace_path = self.tmp_dir_path / "test_builder_merge_backbone"
        tmp_model_path = workspace_path / "model.py"
        with pytest.raises(ValueError):
            self.otx_builder.merge_backbone(tmp_model_path, "unexpected")

    @e2e_pytest_unit
    def test_builder_merge_backbone(self, mocker) -> None:
        """Update model config without backbone's out_indices."""
        mocker.patch("otx.cli.builder.builder.Path.exists", return_value=True)
        mock_backbone_config = {
            "backbone": {"type": "torchvision.resnet18", "use_out_indices": True, "out_indices": [0, 1, 2]}
        }
        mock_mmcv_load = mocker.patch("otx.cli.builder.builder.mmcv.load", return_value=mock_backbone_config)
        mock_model_config = mocker.MagicMock()
        mock_model_config.model.backbone = {"out_indices"}
        mock_config_from_file = mocker.patch(
            "otx.cli.builder.builder.OTXConfig.fromfile", return_value=mock_model_config
        )
        mock_get_backbone_registry = mocker.patch(
            "otx.cli.builder.builder.get_backbone_registry", return_value=(None, ["otx"])
        )
        mock_backbone = mocker.MagicMock()
        mock_build_from_cfg = mocker.patch("otx.cli.builder.builder.build_from_cfg", return_value=mock_backbone)
        mock_get_backbone_out_channels = mocker.patch(
            "otx.cli.builder.builder.get_backbone_out_channels", return_value=[0, 1, 2]
        )
        mock_update_channels = mocker.patch("otx.cli.builder.builder.update_channels", return_value=None)

        Builder().merge_backbone("fake_model_path", "fake_backbone_path")

        mock_mmcv_load.assert_called_once()
        mock_config_from_file.assert_called_once()
        mock_get_backbone_registry.assert_called_once()
        mock_build_from_cfg.assert_called_once()
        mock_get_backbone_out_channels.assert_called_once()
        mock_update_channels.assert_called_once()


class MockBackbone(nn.Module):
    def __init__(self) -> None:
        super(MockBackbone, self).__init__()

    def forward(self, x):
        return x


class MockRegistry(Registry):
    def __init__(self, name: str, parent: Registry = None, scope: str = None) -> None:
        super(MockRegistry, self).__init__(name=name, parent=parent, scope=scope)


class TestOTXBuilderUtils:
    """Check util function in builder.py is working well.

    1. Check "get_backbone_out_channels" function that checking backbone's output channels is working well.
    <Steps>
        1. Check default input_size = 64
        2. Check backbone input_size change to 128

    2. Check "update_backbone_args" function is working.
    <Steps>
        1. Update backbone args (Case without Required Args)
        2. Update required Args in Backbone (Check Missing Args)
        3. Update Args with out_indices
        4. Update backbone using the backbone name from the backbone list (Check updating with options)
        5. Update backbone using the backbone name from the backbone list (Check updating without options)
        6. Raise ValueError with unexpected backbone

    3. Check "update_channels" function is working.
    <Steps>
        1. Remove model.neck.in_channels (GlobalAveragePooling)
        2. Update model.neck.in_channels
        3. Update model.decode_head.in_channels & in_index (segmentation case)
        4. Update model.head.in_channels
        5. Raise NotImplementedError with unexpected model key
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.registry = MockRegistry(name="test")
        self.backbone = MockBackbone

    @e2e_pytest_unit
    def test_get_backbone_out_channels(self) -> None:
        """Check "get_backbone_out_channels" function that checking backbone's output channels is working well.

        1. Check default input_size = 64
        2. Check backbone input_size change to 128
        """

        backbone = MockBackbone()
        expected_result = [64, 64]
        out_channels = get_backbone_out_channels(backbone)
        assert out_channels == expected_result

        backbone.input_size = 128
        expected_result = [128, 128]
        out_channels = get_backbone_out_channels(backbone)
        assert out_channels == expected_result

    @e2e_pytest_unit
    def test_update_backbone_args_without_required_args(self) -> None:
        """Update backbone args (Case without Required Args)."""

        def mock_init(self, a=1, b=2):
            super(MockBackbone, self).__init__()

        self.backbone.__init__ = mock_init
        self.registry.register_module(module=self.backbone)

        backbone_config = {"type": "MockBackbone"}
        inputs = {"backbone_config": backbone_config, "registry": self.registry, "backend": "mmseg"}
        results = update_backbone_args(**inputs)
        expected_results = []
        assert results == expected_results

    @e2e_pytest_unit
    def test_update_backbone_args_required_args(self) -> None:
        """Update required Args in Backbone (Check Missing Args)."""

        def mock_init(self, depth, a=1, b=2):
            super(MockBackbone, self).__init__()

        self.backbone.__init__ = mock_init
        self.registry.register_module(module=self.backbone, force=True)
        backbone_config = {"type": "MockBackbone"}
        inputs = {"backbone_config": backbone_config, "registry": self.registry, "backend": "mmseg"}
        results = update_backbone_args(**inputs)
        expected_results = ["depth"]
        assert results == expected_results

    @e2e_pytest_unit
    def test_update_backbone_args_with_out_indices(self) -> None:
        """Update Args with out_indices."""
        backbone_config = {"type": "MockBackbone", "out_indices": (0, 1, 2, 3)}
        self.registry.register_module(module=self.backbone, force=True)
        inputs = {"backbone_config": backbone_config, "registry": self.registry, "backend": "mmseg"}
        results = update_backbone_args(**inputs)
        expected_results = ["depth"]
        assert results == expected_results
        assert "use_out_indices" in backbone_config

    @e2e_pytest_unit
    def test_update_backbone_args_with_option(self) -> None:
        """Update backbone using the backbone name from the backbone list (Check updating with options)."""
        child_registry = MockRegistry(name="mmseg", parent=self.registry, scope="mmseg")
        backbone_config = {"type": "mmseg.ResNet"}
        child_registry.register_module(name="ResNet", module=self.backbone, force=True)
        inputs = {"backbone_config": backbone_config, "registry": self.registry, "backend": "mmseg"}
        results = update_backbone_args(**inputs)
        expected_results = []
        assert results == expected_results
        expected_depth = 18
        assert "depth" in backbone_config
        assert backbone_config["depth"] == expected_depth

    @e2e_pytest_unit
    def test_update_backbone_args_without_options(self) -> None:
        """Update backbone using the backbone name from the backbone list (Check updating without options)."""

        def mock_init(self, extra):
            super(MockBackbone, self).__init__()

        backbone_config = {"type": "mmseg.HRNet"}
        self.backbone.__init__ = mock_init
        child_registry = MockRegistry(name="mmseg", parent=self.registry, scope="mmseg")
        child_registry.register_module(name="HRNet", module=self.backbone, force=True)
        inputs = {"backbone_config": backbone_config, "registry": self.registry, "backend": "mmseg"}
        results = update_backbone_args(**inputs)
        expected_results = ["extra"]
        assert results == expected_results
        expected_extra_value = "!!!!!!!!!!!INPUT_HERE!!!!!!!!!!!"
        assert "extra" in backbone_config
        assert backbone_config["extra"] == expected_extra_value

    @e2e_pytest_unit
    def test_update_backbone_args_abnormal_backbone_type(self) -> None:
        """Raise ValueError with unexpected backbone."""
        backbone_config = {"type": "unexpected"}
        inputs = {"backbone_config": backbone_config, "registry": self.registry, "backend": "mmseg"}
        with pytest.raises(ValueError):
            update_backbone_args(**inputs)

    @e2e_pytest_unit
    def test_update_channels_neck(self) -> None:
        """Remove model.neck.in_channels."""
        cfg_dict = {"model": {"neck": {"type": "GlobalAveragePooling", "in_channels": 100}}}
        model_config = OTXConfig(cfg_dict=cfg_dict)
        update_channels(model_config, -1)
        assert "in_channels" not in model_config.model.neck

        cfg_dict = {"model": {"neck": {"type": "TestNeck", "in_channels": 100}}}
        model_config = OTXConfig(cfg_dict=cfg_dict)
        update_channels(model_config, -1)
        assert model_config.model.neck.in_channels == -1

    @e2e_pytest_unit
    def test_update_channels_decode_head(self) -> None:
        """Update model.decode_head.in_channels & in_index (segmentation case)."""
        cfg_dict = {"model": {"decode_head": {"in_channels": (0, 1, 2), "in_index": (0, 1, 2)}}}
        out_channels = (10, 20, 30, 40)
        model_config = OTXConfig(cfg_dict=cfg_dict)
        update_channels(model_config, out_channels)
        assert model_config.model.decode_head.in_index == list(range(len(out_channels)))
        assert model_config.model.decode_head.in_channels == out_channels

    @e2e_pytest_unit
    def test_update_channels_head(self) -> None:
        """Update model.head.in_channels."""
        out_channels = (10, 20, 30, 40)
        cfg_dict = {"model": {"head": {"in_channels": (0, 1, 2)}}}
        model_config = OTXConfig(cfg_dict=cfg_dict)
        update_channels(model_config, out_channels)
        assert model_config.model.head.in_channels == out_channels

    @e2e_pytest_unit
    def test_update_channels_abnormal_inputs(self) -> None:
        """Raise NotImplementedError with unexpected model key."""
        out_channels = (10, 20, 30, 40)
        cfg_dict = {"model": {"unexpected": {"in_channels": (0, 1, 2)}}}
        model_config = OTXConfig(cfg_dict=cfg_dict)
        with pytest.raises(NotImplementedError):
            update_channels(model_config, out_channels)
