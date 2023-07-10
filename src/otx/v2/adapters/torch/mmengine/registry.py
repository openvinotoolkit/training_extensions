# Regist OTX custom mmengine modules
from otx.v2.api.core.registry import BaseRegistry

from mmengine.registry import (
    DATA_SAMPLERS,
    DATASETS,
    EVALUATOR,
    FUNCTIONS,
    HOOKS,
    INFERENCERS,
    LOG_PROCESSORS,
    LOOPS,
    METRICS,
    MODEL_WRAPPERS,
    MODELS,
    OPTIM_WRAPPER_CONSTRUCTORS,
    OPTIM_WRAPPERS,
    OPTIMIZERS,
    PARAM_SCHEDULERS,
    RUNNER_CONSTRUCTORS,
    RUNNERS,
    TASK_UTILS,
    TRANSFORMS,
    VISBACKENDS,
    VISUALIZERS,
    WEIGHT_INITIALIZERS,
)

from .modules import *

REGISTRY_LIST = [
    DATA_SAMPLERS,
    DATASETS,
    EVALUATOR,
    FUNCTIONS,
    HOOKS,
    INFERENCERS,
    LOG_PROCESSORS,
    LOOPS,
    METRICS,
    MODEL_WRAPPERS,
    MODELS,
    OPTIM_WRAPPER_CONSTRUCTORS,
    OPTIM_WRAPPERS,
    OPTIMIZERS,
    PARAM_SCHEDULERS,
    RUNNER_CONSTRUCTORS,
    RUNNERS,
    TASK_UTILS,
    TRANSFORMS,
    VISBACKENDS,
    VISUALIZERS,
    WEIGHT_INITIALIZERS,
]


class MMEngineRegistry(BaseRegistry):
    def __init__(self, name="mmengine"):
        super().__init__(name)
        self.module_registry = {registry.name: registry for registry in REGISTRY_LIST}

    def get(self, module_type: str):
        # The module_dict is the highest priority.
        if module_type in self.module_dict:
            return self.module_dict[module_type]

        for module in self.module_registry.values():
            if module_type in module:
                return module.get(module_type)
        return None

    def __repr__(self):
        return self.module_registry
