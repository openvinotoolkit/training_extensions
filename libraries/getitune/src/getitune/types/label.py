# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dataclasses for label information."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

__all__ = [
    "LabelInfo",
    "LabelInfoTypes",
    "NullLabelInfo",
    "SegLabelInfo",
]


@dataclass
class LabelInfo:
    """Object to represent label information."""

    label_names: list[str]
    label_ids: list[str]
    label_groups: list[list[str]]

    @property
    def num_classes(self) -> int:
        """Return number of labels."""
        return len(self.label_names)

    @classmethod
    def from_num_classes(cls, num_classes: int) -> LabelInfo:
        """Create this object from the number of classes.

        Args:
            num_classes: Number of classes

        Returns:
            LabelInfo(
                label_names=["label_0", ...],
                label_groups=[["label_0", ...]]
            )
        """
        if num_classes <= 0:
            return NullLabelInfo()

        label_names = [f"label_{idx}" for idx in range(num_classes)]
        label_ids = [str(i) for i in range(num_classes)]
        return cls(label_names=label_names, label_groups=[label_names], label_ids=label_ids)

    def as_dict(self, normalize_label_names: bool = False) -> dict[str, Any]:
        """Return a dictionary including all params."""
        result = asdict(self)

        if normalize_label_names:

            def normalize_fn(node: str | list | tuple | dict | int) -> str | list | tuple | dict | int:
                """Normalize label names in nested structures."""
                if isinstance(node, str):
                    return node.replace(" ", "_")
                if isinstance(node, list):
                    return [normalize_fn(item) for item in node]
                if isinstance(node, tuple):
                    return tuple(normalize_fn(item) for item in node)
                if isinstance(node, dict):
                    return {normalize_fn(key): normalize_fn(value) for key, value in node.items()}
                return node

            for key in result:
                result[key] = normalize_fn(result[key])

        return result

    def to_json(self) -> str:
        """Return JSON serialized string."""
        return json.dumps(self.as_dict())

    @classmethod
    def from_json(cls, serialized: str) -> LabelInfo:
        """Reconstruct it from the JSON serialized string."""
        labels_info = json.loads(serialized)
        if "label_ids" not in labels_info:
            labels_info["label_ids"] = labels_info["label_names"]
        return cls(**labels_info)


@dataclass
class SegLabelInfo(LabelInfo):
    """Meta information of Semantic Segmentation."""

    ignore_index: int = 255

    @classmethod
    def from_num_classes(cls, num_classes: int) -> SegLabelInfo:
        """Create this object from the number of classes.

        Args:
            num_classes: Number of classes

        Returns:
            LabelInfo(
                label_names=["Background", "label_0", ..., "label_{num_classes - 1}"]
                label_groups=[["Background", "label_0", ..., "label_{num_classes - 1}"]]
            )
        """
        if num_classes == 1:
            label_names = ["background", "label_0"]
            return SegLabelInfo(label_names=label_names, label_groups=[label_names], label_ids=["0", "1"])

        return super().from_num_classes(num_classes)  # type: ignore[return-value]


@dataclass
class NullLabelInfo(LabelInfo):
    """Represent no label information."""

    def __init__(self) -> None:
        super().__init__(label_names=[], label_groups=[[]], label_ids=[])

    @classmethod
    def from_json(cls, _: str) -> LabelInfo:
        """Reconstruct it from the JSON serialized string."""
        return cls()


# Dispatching rules:
# 1. label_info: int => LabelInfo.from_num_classes(label_info)
# 2. label_info: list[str] => LabelInfo(label_names=label_info, label_groups=[label_info])
# 3. label_info: LabelInfo => label_info
LabelInfoTypes = LabelInfo | int | list[str]
