"""UI rules configuration."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


from .rules import NullUIRules, Rule, UIRules
from .types import Action, Operator

__all__ = [
    "NullUIRules",
    "Rule",
    "UIRules",
    "Action",
    "Operator",
]
