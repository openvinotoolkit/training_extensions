"""This module define the Explain entity."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


from dataclasses import dataclass
from typing import Any, Callable, Optional


# pylint: disable=unused-argument
def default_progress_callback(progress: int, score: Optional[float] = None):
    """This is the default progress callback for OptimizationParameters."""


@dataclass
class ExplainParameters:
    """Explain parameters.

    Attributes:
        explainer: Explain algorithm to be used in explanation mode.
            Will be converted automatically to lowercase.
        process_saliency_maps: Processing of saliency map includes (1) resize to input image resolution
            and (2) apply a colormap.
        explain_predicted_classes: Provides explanations only for predicted classes.
            Otherwise, explain all classes.
    """

    update_progress: Callable[[int, Optional[float]], Any] = default_progress_callback

    explainer: str = ""
    process_saliency_maps: bool = False
    explain_predicted_classes: bool = True
