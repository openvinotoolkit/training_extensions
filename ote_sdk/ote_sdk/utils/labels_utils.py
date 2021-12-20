"""
This module implements utilities for labels
"""

# INTEL CONFIDENTIAL
#
# Copyright (C) 2021 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were provided to
# you ("License"). Unless the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit this software or the related documents
# without Intel's prior written permission.
#
# This software and the related documents are provided as is,
# with no express or implied warranties, other than those that are expressly stated
# in the License.

from typing import List, Optional

from ote_sdk.entities.label import LabelEntity
from ote_sdk.entities.label_schema import LabelSchemaEntity
from ote_sdk.entities.scored_label import ScoredLabel


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


def get_leaf_labels(label_schema: LabelSchemaEntity) -> List[LabelEntity]:
    """
    Get leafs from label tree
    """
    leaf_labels = []
    all_labels = label_schema.get_labels(False)
    for lbl in all_labels:
        if not label_schema.get_children(lbl):
            leaf_labels.append(lbl)

    return leaf_labels


def get_ancestors_by_prediction(
    label_schema: LabelSchemaEntity, prediction: ScoredLabel
) -> List[ScoredLabel]:
    """
    Get all the ancestors for a given label node
    """
    ancestor_labels = label_schema.get_ancestors(prediction.get_label())
    return [ScoredLabel(al, prediction.probability) for al in ancestor_labels]
