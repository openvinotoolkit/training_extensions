"""This module implements the AnalyseParameters entity."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


from dataclasses import dataclass
from typing import Any, Callable, Optional


# pylint: disable=unused-argument
def default_progress_callback(progress: int, score: Optional[float] = None):
    """This is the default progress callback for OptimizationParameters."""


@dataclass
class InferenceParameters:
    """Inference parameters.

    Attributes:
        is_evaluation: Set to ``True`` if the output dataset is intended
            to be used for evaluation purposes. In this scenario, any
            postprocessing filtering (such as thresholding and NMS)
            should be disabled to avoid interfering with algorithms such
            as NMS.
        update_progress: Callback which can be used to provide updates
            about the progress of a task.
        explainer: Explain algorithm to be used in explanation mode.
            Will be converted automatically to lowercase.
        process_saliency_maps: Process saliency map to input image resolution and apply colormap
        explain_predicted_classes: If set to True, provide explanations only for predicted classes.
            Otherwise, explain all classes.
    """

    is_evaluation: bool = False
    update_progress: Callable[[int, Optional[float]], Any] = default_progress_callback

    # TODO(negvet): use separate ExplainParameters dataclass for this
    explainer: str = ""
    process_saliency_maps: bool = False
    explain_predicted_classes: bool = True
