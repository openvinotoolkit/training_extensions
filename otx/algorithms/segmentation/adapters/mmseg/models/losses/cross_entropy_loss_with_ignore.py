"""Cross entropy loss for ignored mode in class-incremental learning."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn.functional as F
from mmseg.models.builder import LOSSES
from mmseg.models.losses import CrossEntropyLoss
from mmseg.models.losses.utils import get_class_weight


@LOSSES.register_module()
class CrossEntropyLossWithIgnore(CrossEntropyLoss):
    """CrossEntropyLossWithIgnore with Ignore Mode Support for Class Incremental Learning.

    Args:
        model_classes (list[str]): Model classes
        bg_aware (bool, optional): Whether to enable BG-aware loss
            'background' class would be added the start of model classes/label schema
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
