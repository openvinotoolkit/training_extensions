"""Unit-Test Case for otx.cli.builder.builder."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os

import mmcv
import pytest
from mmcv.utils import Registry
from torch import nn

from otx.cli.builder.builder import (
    Builder,
    get_backbone_out_channels,
    update_backbone_args,
    update_channels,
)
from otx.cli.utils.importing import get_otx_root_path
from otx.mpa.utils.config_utils import MPAConfig
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestOTXCLIBuilder:
    """Check Builder's function is working well.

    1. Check "Builder.build_task_config" function that create otx-workspace is working well.
    <Steps>
        1. Create Classification custom workspace
        2. Raising Error of building workspace with already created path
        3. Update hparam.yaml with train_type="selfsl"
        4. Raising ValueError with wrong train_type
        5. Build workspace with model_type argments
        6. Raise ValueError when build workspace with wrong model_type argments

    2. Check "Builder.build_backbone_config" function that generate backbone configuration file is working well
    <Steps>
        1. Generate backbone config file (mmcls.MMOVBackbone)
        2. Raise ValueError with wrong output_path

    3. Check "Builder.build_model_config" function that update model config with new backbone is working well
    <Steps>
        1. Update model config with mmcls.ResNet backbone (default model.backbone: otx.OTXEfficientNet)
        2. Raise ValueError with wrong model_config_path
        3. Raise ValueError with wrong backbone_config_path
        4. Update model config without backbone's out_indices
        5. Update model config with backbone's pretrained path
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir_path):
        self.otx_builder = Builder()
        self.otx_root = get_otx_root_path()
        self.tmp_dir_path = tmp_dir_path

    @e2e_pytest_unit
    def test_builder_build_task_config_create_workspace(self):
        """Create Classification custom workspace."""
        workspace_path = os.path.join(self.tmp_dir_path, "test_builder_build_task_config_create_workspace")
        inputs = {"task_type": "classification", "workspace_path": workspace_path, "otx_root": self.otx_root}
        self.otx_builder.build_task_config(**inputs)
        assert os.path.exists(workspace_path)
        assert os.path.exists(os.path.join(workspace_path, "configuration.yaml"))
        assert os.path.exists(os.path.join(workspace_path, "template.yaml"))
        assert os.path.exists(os.path.join(workspace_path, "model.py"))
        assert os.path.exists(os.path.join(workspace_path, "data.yaml"))
        assert os.path.exists(os.path.join(workspace_path, "data_pipeline.py"))

    @e2e_pytest_unit
    def test_builder_build_task_config_reuse_same_path(self):
        """Raising Error of building workspace with already created path."""
        workspace_path = os.path.join(self.tmp_dir_path, "test_builder_build_task_config_create_workspace")
        inputs = {"task_type": "classification", "workspace_path": workspace_path, "otx_root": self.otx_root}
        with pytest.raises(FileExistsError):
            self.otx_builder.build_task_config(**inputs)

    @e2e_pytest_unit
    def test_builder_build_task_config_normal_train_type(self):
        """Update hparam.yaml with train_type="selfsl"."""
        workspace_path = os.path.join(self.tmp_dir_path, "test_builder_build_task_config_check_update_hparams")
        train_type = "selfsl"
        inputs = {
            "task_type": "classification",
            "train_type": train_type,
            "workspace_path": workspace_path,
            "otx_root": self.otx_root,
        }
        self.otx_builder.build_task_config(**inputs)
        assert os.path.exists(workspace_path)
        assert os.path.exists(os.path.join(workspace_path, "configuration.yaml"))
        assert os.path.exists(os.path.join(workspace_path, "template.yaml"))
        assert os.path.exists(os.path.join(workspace_path, "data.yaml"))
        template = MPAConfig.fromfile(os.path.join(workspace_path, "template.yaml"))
        expected_template_train_type = {"default_value": "SELFSUPERVISED"}
        assert template.hyper_parameters.parameter_overrides.algo_backend.train_type == expected_template_train_type
        model_dir = os.path.join(workspace_path, train_type)
        assert os.path.exists(model_dir)
        assert os.path.exists(os.path.join(model_dir, "model.py"))
        assert os.path.exists(os.path.join(model_dir, "data_pipeline.py"))

    @e2e_pytest_unit
    def test_builder_build_task_config_abnormal_train_type(self):
        """Raising ValueError with wrong train_type."""
        workspace_path = os.path.join(self.tmp_dir_path, "test_builder_build_task_config_abnormal_train_type")
        train_type = "unexpected"
        inputs = {
            "task_type": "classification",
            "train_type": train_type,
            "workspace_path": workspace_path,
            "otx_root": self.otx_root,
        }
        with pytest.raises(ValueError):
            self.otx_builder.build_task_config(**inputs)

    @e2e_pytest_unit
    def test_builder_build_task_config_normal_model_type(self):
        """Build workspace with model_type argments."""
        workspace_path = os.path.join(self.tmp_dir_path, "test_builder_build_task_config_normal_model_type")
        model_type = "yolox"
        inputs = {
            "task_type": "detection",
            "model_type": model_type,
            "workspace_path": workspace_path,
            "otx_root": self.otx_root,
        }
        self.otx_builder.build_task_config(**inputs)
        assert os.path.exists(os.path.join(workspace_path, "template.yaml"))
        template = MPAConfig.fromfile(os.path.join(workspace_path, "template.yaml"))
        assert template.name.lower() == model_type

    @e2e_pytest_unit
    def test_builder_build_task_config_abnormal_model_type(self):
        """Raise ValueError when build workspace with wrong model_type argments."""
        workspace_path = os.path.join(self.tmp_dir_path, "test_builder_build_task_config_abnormal_model_type")
        inputs = {
            "task_type": "detection",
            "model_type": "unexpected",
            "workspace_path": workspace_path,
            "otx_root": self.otx_root,
        }
        with pytest.raises(ValueError):
            self.otx_builder.build_task_config(**inputs)

    @e2e_pytest_unit
    @pytest.mark.parametrize("backbone_type", ["mmcls.MMOVBackbone"])
    def test_builder_build_backbone_config_generate_backbone(self, backbone_type):
        """Generate backbone config file (mmcls.MMOVBackbone)."""
        tmp_backbone_path = os.path.join(self.tmp_dir_path, "backbone.yaml")
        self.otx_builder.build_backbone_config(backbone_type, tmp_backbone_path)
        assert os.path.exists(tmp_backbone_path)
        backbone_config = mmcv.load(tmp_backbone_path)
        assert backbone_config["backbone"]["type"] == backbone_type

    @e2e_pytest_unit
    @pytest.mark.parametrize("backbone_type", ["mmcls.MMOVBackbone"])
    def test_builder_build_backbone_config_abnormal_output_path(self, backbone_type):
        """Raise ValueError with wrong output_path."""
        tmp_backbone_path = os.path.join(self.tmp_dir_path, "wrong.path")
        with pytest.raises(ValueError):
            self.otx_builder.build_backbone_config(backbone_type, tmp_backbone_path)

    @e2e_pytest_unit
    @pytest.mark.parametrize("backbone_type", ["mmcls.ResNet"])
    def test_builder_build_model_config_update_model_config(self, backbone_type):
        """Update model config with mmcls.ResNet backbone (default model.backbone: otx.OTXEfficientNet)."""
        workspace_path = os.path.join(self.tmp_dir_path, "test_builder_build_model_config")
        inputs = {"task_type": "classification", "workspace_path": workspace_path, "otx_root": self.otx_root}
        self.otx_builder.build_task_config(**inputs)
        tmp_model_path = os.path.join(workspace_path, "model.py")
        assert os.path.exists(tmp_model_path)
        pre_model_config = MPAConfig.fromfile(tmp_model_path)
        assert pre_model_config.model.backbone.type == "otx.OTXEfficientNet"

        tmp_backbone_path = os.path.join(workspace_path, "backbone.yaml")
        self.otx_builder.build_backbone_config(backbone_type, tmp_backbone_path)
        assert os.path.exists(tmp_backbone_path)

        self.otx_builder.build_model_config(tmp_model_path, tmp_backbone_path)
        assert os.path.exists(tmp_model_path)
        updated_model_config = MPAConfig.fromfile(tmp_model_path)
        assert updated_model_config.model.backbone.type == backbone_type

    @e2e_pytest_unit
    def test_builder_build_model_config_abnormal_model_path(self):
        """Raise ValueError with wrong model_config_path."""
        workspace_path = os.path.join(self.tmp_dir_path, "test_builder_build_model_config")
        tmp_backbone_path = os.path.join(workspace_path, "backbone.yaml")
        with pytest.raises(ValueError):
            self.otx_builder.build_model_config("unexpected", tmp_backbone_path)

    @e2e_pytest_unit
    def test_builder_build_model_config_abnormal_backbone_path(self):
        """Raise ValueError with wrong backbone_config_path."""
        workspace_path = os.path.join(self.tmp_dir_path, "test_builder_build_model_config")
        tmp_model_path = os.path.join(workspace_path, "model.py")
        with pytest.raises(ValueError):
            self.otx_builder.build_model_config(tmp_model_path, "unexpected")

    @e2e_pytest_unit
    def test_builder_build_model_config_without_out_indices(self):
        """Update model config without backbone's out_indices."""
        workspace_path = os.path.join(self.tmp_dir_path, "test_builder_build_model_config")
        tmp_model_path = os.path.join(workspace_path, "model.py")
        tmp_backbone_path = os.path.join(workspace_path, "backbone.yaml")
        backbone_config = mmcv.load(tmp_backbone_path)
        assert backbone_config["backbone"].pop("out_indices") == (3,)
        mmcv.dump(backbone_config, tmp_backbone_path)
        self.otx_builder.build_model_config(tmp_model_path, tmp_backbone_path)
        updated_model_config = MPAConfig.fromfile(tmp_model_path)
        assert "out_indices" in updated_model_config.model.backbone

    @e2e_pytest_unit
    def test_builder_build_model_config_backbone_pretrained(self):
        """Update model config with backbone's pretrained path."""
        workspace_path = os.path.join(self.tmp_dir_path, "test_builder_build_model_config")
        tmp_model_path = os.path.join(workspace_path, "model.py")
        tmp_backbone_path = os.path.join(workspace_path, "backbone.yaml")
        backbone_config = mmcv.load(tmp_backbone_path)
        expected_pretrained = "pretrained.path"
        backbone_config["backbone"]["pretrained"] = expected_pretrained
        mmcv.dump(backbone_config, tmp_backbone_path)
        self.otx_builder.build_model_config(tmp_model_path, tmp_backbone_path)
        updated_model_config = MPAConfig.fromfile(tmp_model_path)
        assert updated_model_config.model.pretrained == expected_pretrained


class MockBackbone(nn.Module):
    def __init__(self):
        super(MockBackbone, self).__init__()

    def forward(self, x):
        return x


class MockRegistry(Registry):
    def __init__(self, name, parent=None, scope=None):
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
    def setup(self):
        self.registry = MockRegistry(name="test")
        self.backbone = MockBackbone

    @e2e_pytest_unit
    def test_get_backbone_out_channels(self):
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
    def test_update_backbone_args_without_required_args(self):
        """Update backbone args (Case without Required Args)."""

        def mock_init(self, a=1, b=2):
            super(MockBackbone, self).__init__()

        self.backbone.__init__ = mock_init
        self.registry.register_module(module=self.backbone)

        backbone_config = {
            "type": "MockBackbone",
        }
        inputs = {"backbone_config": backbone_config, "registry": self.registry, "backend": "mmseg"}
        results = update_backbone_args(**inputs)
        expected_results = []
        assert results == expected_results

    @e2e_pytest_unit
    def test_update_backbone_args_required_args(self):
        """Update required Args in Backbone (Check Missing Args)."""

        def mock_init(self, depth, a=1, b=2):
            super(MockBackbone, self).__init__()

        self.backbone.__init__ = mock_init
        self.registry.register_module(module=self.backbone, force=True)
        backbone_config = {
            "type": "MockBackbone",
        }
        inputs = {"backbone_config": backbone_config, "registry": self.registry, "backend": "mmseg"}
        results = update_backbone_args(**inputs)
        expected_results = ["depth"]
        assert results == expected_results

    @e2e_pytest_unit
    def test_update_backbone_args_with_out_indices(self):
        """Update Args with out_indices."""
        backbone_config = {"type": "MockBackbone", "out_indices": (0, 1, 2, 3)}
        self.registry.register_module(module=self.backbone, force=True)
        inputs = {"backbone_config": backbone_config, "registry": self.registry, "backend": "mmseg"}
        results = update_backbone_args(**inputs)
        expected_results = ["depth"]
        assert results == expected_results
        assert "use_out_indices" in backbone_config

    @e2e_pytest_unit
    def test_update_backbone_args_with_option(self):
        """Update backbone using the backbone name from the backbone list (Check updating with options)."""
        child_registry = MockRegistry(name="mmseg", parent=self.registry, scope="mmseg")
        backbone_config = {
            "type": "mmseg.ResNet",
        }
        child_registry.register_module(name="ResNet", module=self.backbone, force=True)
        inputs = {"backbone_config": backbone_config, "registry": self.registry, "backend": "mmseg"}
        results = update_backbone_args(**inputs)
        expected_results = []
        assert results == expected_results
        expected_depth = 18
        assert "depth" in backbone_config
        assert backbone_config["depth"] == expected_depth

    @e2e_pytest_unit
    def test_update_backbone_args_without_options(self):
        """Update backbone using the backbone name from the backbone list (Check updating without options)."""

        def mock_init(self, extra):
            super(MockBackbone, self).__init__()

        backbone_config = {
            "type": "mmseg.HRNet",
        }
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
    def test_update_backbone_args_abnormal_backbone_type(self):
        """Raise ValueError with unexpected backbone."""
        backbone_config = {
            "type": "unexpected",
        }
        inputs = {"backbone_config": backbone_config, "registry": self.registry, "backend": "mmseg"}
        with pytest.raises(ValueError):
            update_backbone_args(**inputs)

    @e2e_pytest_unit
    def test_update_channels_neck(self):
        """Remove model.neck.in_channels."""
        cfg_dict = {"model": {"neck": {"type": "GlobalAveragePooling", "in_channels": 100}}}
        model_config = MPAConfig(cfg_dict=cfg_dict)
        update_channels(model_config, -1)
        assert "in_channels" not in model_config.model.neck

        cfg_dict = {"model": {"neck": {"type": "TestNeck", "in_channels": 100}}}
        model_config = MPAConfig(cfg_dict=cfg_dict)
        update_channels(model_config, -1)
        assert model_config.model.neck.in_channels == -1

    @e2e_pytest_unit
    def test_update_channels_decode_head(self):
        """Update model.decode_head.in_channels & in_index (segmentation case)."""
        cfg_dict = {"model": {"decode_head": {"in_channels": (0, 1, 2), "in_index": (0, 1, 2)}}}
        out_channels = (10, 20, 30, 40)
        model_config = MPAConfig(cfg_dict=cfg_dict)
        update_channels(model_config, out_channels)
        assert model_config.model.decode_head.in_index == list(range(len(out_channels)))
        assert model_config.model.decode_head.in_channels == out_channels

    @e2e_pytest_unit
    def test_update_channels_head(self):
        """Update model.head.in_channels."""
        out_channels = (10, 20, 30, 40)
        cfg_dict = {"model": {"head": {"in_channels": (0, 1, 2)}}}
        model_config = MPAConfig(cfg_dict=cfg_dict)
        update_channels(model_config, out_channels)
        assert model_config.model.head.in_channels == out_channels

    @e2e_pytest_unit
    def test_update_channels_abnormal_inputs(self):
        """Raise NotImplementedError with unexpected model key."""
        out_channels = (10, 20, 30, 40)
        cfg_dict = {"model": {"unexpected": {"in_channels": (0, 1, 2)}}}
        model_config = MPAConfig(cfg_dict=cfg_dict)
        with pytest.raises(NotImplementedError):
            update_channels(model_config, out_channels)
