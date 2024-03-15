"""This module contains the ModelLifecycle Enum."""


# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from enum import Enum, auto


class ModelLifecycle(Enum):
    """This Enum represents the different stages in the ModelLifecycle.

    It is used by configuration parameters to indicate
    in which stage of the model lifecycle the parameter takes effect. Selecting a stage early in the lifecycle implies
    that all downstream stages are affected as well (e.g. if this is set to `ModelLifecycle.TRAINING`, it is assumed
    that inference and testing are also impacted).

    Currently the following stages are possible:
    ARCHITECTURE - Select this stage if the parameter modifies the model architecture, such that the most recently
        trained weights cannot directly by used for the next training round due to a model topology mismatch. For
        example, a parameter `model_depth` that controls the number of downsampling steps in a UNet model should
        have this stage set.
    TRAINING - Select this stage if the parameter is likely to change the outcome of the training process. For example,
        the parameter `learning_rate` should have this stage set.
    INFERENCE - Select this stage if the parameter changes the result of inference. For example, a parameter
        `probability_threshold` that controls the threshold for binary classification should have this stage set.
    TESTING - Select this stage if the parameter changes the outcome of the evaluation process. For example, a parameter
        'test_metric` that controls which metric to use for testing does not change training or inference results, but
        does affect the final evaluation of the model. Therefore, it should have this stage set.
    NONE - Select this stage if the parameter is non-functional, for example if it only impacts training speed but
        should not change the training outcome. For example, a parameter `num_workers` that controls the number of
        threads used in a dataloader should have this stage set.
    """

    NONE = auto()
    ARCHITECTURE = auto()
    TRAINING = auto()
    INFERENCE = auto()
    TESTING = auto()

    def __str__(self):
        """Retrieves the string representation of an instance of the Enum."""
        return self.name
