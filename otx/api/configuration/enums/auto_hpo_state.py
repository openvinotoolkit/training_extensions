"""This module contains the AutoHPOState Enum."""


# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from enum import Enum


class AutoHPOState(Enum):
    """Holds metadata related to automatic hyper parameter optimization (auto-HPO) for a single configurable parameter.

    It contains the following values:
        NOT_POSSIBLE  - This implies that the parameter cannot be optimized via auto-HPO
        POSSIBLE      - This implies that the parameter can potentially be optimized via
                        auto-HPO, but auto-HPO has not been carried out for this parameter
                        yet
        OPTIMIZED     - This implies that the parameter has been optimized via auto-HPO,
                        such that the current value of the parameter reflects it's optimal
                        value
        OVERRIDDEN    - This implies that the parameter has previously been optimized via
                        auto-HPO, but it's value has been manually overridden
    """

    NOT_POSSIBLE = "not_possible"
    POSSIBLE = "possible"
    OPTIMIZED = "optimized"
    OVERRIDDEN = "overridden"

    def __str__(self):
        """Retrieves the string representation of an instance of the Enum."""
        return self.value
