# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""X3DFastRCNN model implementation."""

from otx.core.model.entity.action_detection import MMActionCompatibleModel

_MM_CONFIG = """

"""


class X3DFastRCNN(MMActionCompatibleModel):
    """X3D Model."""

    MM_CONFIG = _MM_CONFIG
