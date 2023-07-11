"""Sorting-maximization-related modules for otx.core.ov.ops."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field

from otx.core.ov.ops.builder import OPS
from otx.core.ov.ops.op import Attribute, Operation


@dataclass
class TopKV3Attribute(Attribute):
    """TopKV3Attribute class."""

    axis: int
    mode: str
    sort: str
    index_element_type: str = field(default="i32")

    def __post_init__(self):
        """TopKV3Attribute's post-init function."""
        super().__post_init__()
        valid_mode = ["min", "max"]
        if self.mode not in valid_mode:
            raise ValueError(f"Invalid mode {self.mode}. " f"It must be one of {valid_mode}.")

        valid_sort = ["value", "index", "none"]
        if self.sort not in valid_sort:
            raise ValueError(f"Invalid sort {self.sort}. " f"It must be one of {valid_sort}.")

        valid_index_element_type = ["i32", "i64"]
        if self.index_element_type not in valid_index_element_type:
            raise ValueError(
                f"Invalid index_element_type {self.index_element_type}. "
                f"It must be one of {valid_index_element_type}."
            )


@OPS.register()
class TopKV3(Operation[TopKV3Attribute]):
    """TopKV3 class."""

    TYPE = "TopK"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = TopKV3Attribute

    def forward(self, inputs, k):
        """TopKV3's forward function."""
        raise NotImplementedError


@dataclass
class NonMaxSuppressionV5Attribute(Attribute):
    """NonMaxSuppressionV5Attribute class."""

    box_encoding: str = field(default="corner")
    sort_result_descending: bool = field(default=True)
    output_type: str = field(default="i64")


@OPS.register()
class NonMaxSuppressionV5(Operation[NonMaxSuppressionV5Attribute]):
    """NonMaxSuppressionV5 class."""

    TYPE = "NonMaxSuppression"
    VERSION = "opset5"
    ATTRIBUTE_FACTORY = NonMaxSuppressionV5Attribute

    def forward(
        self,
        boxes,
        scores,
        max_output_boxes_per_class,
        iou_threshold=0,
        score_threshold=0,
        soft_nms_sigma=0,
    ):
        """NonMaxSuppressionV5's forward function."""
        raise NotImplementedError


@dataclass
class NonMaxSuppressionV9Attribute(Attribute):
    """NonMaxSuppressionV9Attribute class."""

    box_encoding: str = field(default="corner")
    sort_result_descending: bool = field(default=True)
    output_type: str = field(default="i64")


@OPS.register()
class NonMaxSuppressionV9(Operation[NonMaxSuppressionV9Attribute]):
    """NonMaxSuppressionV9 class."""

    TYPE = "NonMaxSuppression"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = NonMaxSuppressionV9Attribute

    def forward(
        self,
        boxes,
        scores,
        max_output_boxes_per_class,
        iou_threshold=0,
        score_threshold=0,
        soft_nms_sigma=0,
    ):
        """NonMaxSuppressionV9's forward function."""
        raise NotImplementedError
