"""Loss list of mmdetection adapters."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .cross_focal_loss import CrossSigmoidFocalLoss, OrdinaryFocalLoss

__all__ = ["CrossSigmoidFocalLoss", "OrdinaryFocalLoss"]
