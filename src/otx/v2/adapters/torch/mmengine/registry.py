# Regist OTX custom mmengine modules
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

from otx.v2.api.core.registry import BaseRegistry

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
        self._registry_dict = {registry.name: registry for registry in REGISTRY_LIST}
