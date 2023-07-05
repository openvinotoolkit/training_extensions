# Regist OTX custom mmcv modules
from otx.v2.api.core.registry import BaseRegistry

from mmcv.cnn import MODELS
from mmcv.runner import HOOKS, OPTIMIZERS, RUNNERS

from .modules import *


class MMCVRegistry(BaseRegistry):
    def __init__(self, name="mmcv"):
        super().__init__(name)
        self.module_registry = {"models": MODELS, "optimizers": OPTIMIZERS, "runners": RUNNERS, "hooks": HOOKS}

    def get(self, module_type: str):
        # The module_dict is the highest priority.
        if module_type in self.module_dict:
            return self.module_dict[module_type]

        for module in self.module_registry.values():
            if module_type in module:
                return module.get(module_type)
        return None
