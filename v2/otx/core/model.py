import os
from abc import abstractmethod
from enum import IntEnum
from typing import Dict, Union

from otx.utils.config import Config
from otx.utils.logger import get_logger

logger = get_logger()


class ModelStatus(IntEnum):
    CONFIGURED = 0
    BUILT = 1
    CONFIG_UPDATED = 2
    TRAINED = 3
    OPTIMIZED = 4


class ModelSpec(IntEnum):
    Classifier = 0
    Detector = 1
    Segmentor = 2


class IModel:
    def __init__(self, model_config: Union[Dict, str]):
        if isinstance(model_config, Dict):
            self._config = Config(model_config)
        elif isinstance(model_config, str):
            if os.path.exists(model_config):
                self._config = Config.fromfile(model_config)
            else:
                raise RuntimeError(f"Cannot find configuration file {model_config}")
        else:
            raise RuntimeError(f"Not supported model configuration type [{type(model_config)}]")
        self.spec = self._config.pop("spec")
        self._ckpt = None

    @abstractmethod
    def save(self, model, path, **kwargs):
        """"""
        raise NotImplementedError()

    @abstractmethod
    def export(self, model, path, exp_type="openvion"):
        """"""
        raise NotImplementedError()

    @abstractmethod
    def build(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def update_config(self, config: dict):
        raise NotImplementedError()

    @property
    def config(self):
        return self._config

    @property
    def ckpt(self):
        if self._ckpt is not None:
            if not os.path.exists(self._ckpt):
                logger.warning(f"invalid model checkpoint path: {self._ckpt}")
        return self._ckpt

    @ckpt.setter
    def ckpt(self, value):
        self._ckpt = value
