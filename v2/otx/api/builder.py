import os

from otx import OTXConstants
from otx.utils.config import Config
from otx.utils.logger import get_logger

logger = get_logger()


def build_model_config(model_yaml: str, backbone_yaml: str) -> str:
    if not os.path.exists(model_yaml):
        raise FileExistsError(f"cannot find model configuration file {model_yaml}")
    if not os.path.exists(backbone_yaml):
        raise FileExistsError(f"cannot find backbone configuration file {backbone_yaml}")
    model_cfg = Config.fromfile(model_yaml)
    logger.info(f"model cfg = {model_cfg}")
    backbone_cfg = Config.fromfile(backbone_yaml)
    logger.info(f"backbone cfg = {backbone_cfg}")

    model_cfg.merge_from_dict(backbone_cfg._cfg_dict)
    logger.info(f"updated model cfg = {model_cfg}")

    # TODO: dump new model configuration to the specific target path
    new_model_yaml = os.path.join(OTXConstants.TEMP_PATH, "new_model.yaml")
    model_cfg.dump(new_model_yaml)
    return new_model_yaml


def build_task_config(task_yaml, model_yaml) -> str:
    if not os.path.exists(task_yaml):
        raise FileExistsError(f"cannot find task configuration file {task_yaml}")
    if not os.path.exists(model_yaml):
        raise FileExistsError(f"cannot find model configuration file {model_yaml}")
    task_cfg = Config.fromfile(task_yaml)
    logger.info(f"task cfg = {task_cfg}")
    task_cfg.model.default = model_yaml
    logger.info(f"updated task cfg = {task_cfg}")

    # TODO: dump new task configuration to the specific target path
    new_task_yaml = os.path.join(OTXConstants.TEMP_PATH, "new_task.yaml")
    task_cfg.dump(new_task_yaml)
    return new_task_yaml
