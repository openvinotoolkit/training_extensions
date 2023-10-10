"""OTX adapters.torch.mmengine.Registry module."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


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
    """A registry for registering and retrieving MMEngine modules.

    Attributes:
    ----------
        name (str): The name of the registry.
    """

    def __init__(self, name: str = "mmengine") -> None:
        """Initialize a new instance of the MMEngineRegistry class.

        Args:
        ----
            name (str): The name of the registry. Defaults to "mmengine".
        """
        super().__init__(name)
        self._registry_dict = {registry.name: registry for registry in REGISTRY_LIST}
