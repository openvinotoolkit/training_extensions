"""OTX adapters.torch.mmengine.mmseg.Registry module."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mmseg.registry import (
    DATA_SAMPLERS,
    DATASETS,
    HOOKS,
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

from otx.v2.adapters.torch.mmengine.registry import MMEngineRegistry

REGISTRY_LIST = [
    DATA_SAMPLERS,
    DATASETS,
    HOOKS,
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


class MMSegmentationRegistry(MMEngineRegistry):
    """Registry for MMSegmentation models and related components."""

    def __init__(self, name: str = "mmseg") -> None:
        """Initialize a new instance of the `Registry` class.

        Args:
            name (str): The name of the registry. Defaults to "mmseg".
        """
        super().__init__(name)
        self._registry_dict = {registry.name: registry for registry in REGISTRY_LIST}
