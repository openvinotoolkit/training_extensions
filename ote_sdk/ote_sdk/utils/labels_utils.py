"""
This module implements utilities for labels
"""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Optional

from ote_sdk.entities.label import LabelEntity
from ote_sdk.entities.label_schema import LabelSchemaEntity


def get_empty_label(label_schema: LabelSchemaEntity) -> Optional[LabelEntity]:
    """
    Get first empty label from label_schema
    """
    empty_candidates = list(
        set(label_schema.get_labels(include_empty=True))
        - set(label_schema.get_labels(include_empty=False))
    )
    if empty_candidates:
        return empty_candidates[0]
    return None
