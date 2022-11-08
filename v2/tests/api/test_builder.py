import os

import pytest

from otx import OTXConstants
from otx.api.builder import build_model_config, build_task_config
from otx.utils.config import Config


def test_build_model_config():
    model_yaml = os.path.join(OTXConstants.CONFIG_PATH, "models/classification/image_classifier.yaml")
    backbone_yaml = os.path.join(OTXConstants.CONFIG_PATH, "models/backbones/mobilenet_v3.yaml")

    with pytest.raises(FileExistsError):
        build_model_config("path/not/exists", "path/not/exists")
    with pytest.raises(FileExistsError):
        build_model_config(model_yaml, "path/not/exists")

    new_model_yaml = build_model_config(model_yaml, backbone_yaml)
    assert new_model_yaml is not None
    print(f"new model yaml = {new_model_yaml}")
    assert isinstance(new_model_yaml, str)
    assert os.path.exists(new_model_yaml)

    new_model_cfg = Config.fromfile(new_model_yaml)
    print(f"new model cfg = {new_model_cfg}")
    backbone_cfg = Config.fromfile(backbone_yaml)
    assert new_model_cfg.backbone.type == backbone_cfg.backbone.type


def test_build_task_config():
    task_yaml = os.path.join(OTXConstants.CONFIG_PATH, "tasks/classification/cls-incr-classifier.yaml")
    model_yaml = os.path.join(OTXConstants.CONFIG_PATH, "models/classification/sam_image_classifier.yaml")

    with pytest.raises(FileExistsError):
        build_task_config("path/not/exists", "path/not/exists")
    with pytest.raises(FileExistsError):
        build_task_config(task_yaml, "path/not/exists")

    new_task_yaml = build_task_config(task_yaml, model_yaml)
    assert new_task_yaml is not None
    assert isinstance(new_task_yaml, str)
    assert os.path.exists(new_task_yaml)

    new_task_cfg = Config.fromfile(new_task_yaml)
    print(f"new task cfg = {new_task_cfg}")
    assert new_task_cfg.model.default == model_yaml
