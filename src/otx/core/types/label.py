# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Dataclasses for label information."""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datumaro import Label, LabelCategories

__all__ = ["LabelInfo", "HLabelInfo", "SegLabelInfo", "NullLabelInfo"]


@dataclass
class LabelInfo:
    """Object to represent label information."""

    label_names: list[str]
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
        label_names = [f"label_{idx}" for idx in range(num_classes)]

        return LabelInfo(
            label_names=label_names,
            label_groups=[label_names],
        )

    @classmethod
    def from_dm_label_groups(cls, dm_label_categories: LabelCategories) -> LabelInfo:
        """Create this object from the datumaro label groups.

        Args:
            dm_label_categories (LabelCategories): The label category information from Datumaro.

        Returns:
            LabelInfo(
                label_names=["Heart_King", "Heart_Queen", "Spade_King", "Spade_Jack"]
                label_groups=[["Heart_King", "Heart_Queen"], ["Spade_King", "Spade_Jack"]]
            )

        """
        label_names = [item.name for item in dm_label_categories.items]
        label_groups = [label_group.labels for label_group in dm_label_categories.label_groups]
        if len(label_groups) == 0:  # Single-label classification
            label_groups = [label_names]

        return LabelInfo(
            label_names=label_names,
            label_groups=label_groups,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a dictionary including all params."""
        return asdict(self)

    def to_json(self) -> str:
        """Return JSON serialized string."""
        return json.dumps(self.as_dict())

    @classmethod
    def from_json(cls, serialized: str) -> LabelInfo:
        """Reconstruct it from the JSON serialized string."""
        return cls(**json.loads(serialized))


@dataclass
class HLabelInfo(LabelInfo):
    """The label information represents the hierarchy.

    All params should be kept since they're also used at the Model API side.

    :param num_multiclass_heads: the number of the multiclass heads
    :param num_multilabel_classes: the number of multilabel classes
    :param head_to_logits_range: the logit range of each heads
    :param num_single_label_classes: the number of single label classes
    :param class_to_group_idx: represents the head index and label index
    :param all_groups: represents information of all groups
    :param label_to_idx: index of each label
    :param empty_multiclass_head_indices: the index of head that doesn't include any label
                                          due to the label removing

    i.e.
    Single-selection group information (Multiclass, Exclusive)
    {
        "Shape": ["Rigid", "Non-Rigid"],
        "Rigid": ["Rectangle", "Triangle"],
        "Non-Rigid": ["Circle"]
    }

    Multi-selection group information (Multilabel)
    {
        "Animal": ["Lion", "Panda"]
    }

    In the case above, HlabelInfo will be generated as below.
    NOTE, If there was only one label in the multiclass group, it will be handeled as multilabel(Circle).

        num_multiclass_heads: 2  (Shape, Rigid)
        num_multilabel_classes: 3 (Circle, Lion, Panda)
        head_to_logits_range: {'0': (0, 2), '1': (2, 4)} (Each multiclass head have 2 labels)
        num_single_label_classes: 4 (Rigid, Non-Rigid, Rectangle, Triangle)
        class_to_group_idx: {
            'Non-Rigid': (0, 0), 'Rigid': (0, 1),
            'Rectangle': (1, 0), 'Triangle': (1, 1),
            'Circle': (2, 0), 'Lion': (2,1), 'Panda': (2,2)
        } (head index, label index for each head)
        all_groups: [['Non-Rigid', 'Rigid'], ['Rectangle', 'Triangle'], ['Circle'], ['Lion'], ['Panda']]
        label_to_idx: {
            'Rigid': 0, 'Rectangle': 1,
            'Triangle': 2, 'Non-Rigid': 3, 'Circle': 4
            'Lion': 5, 'Panda': 6
        }
        label_tree_edges: [
            ["Rectangle", "Rigid"], ["Triangle", "Rigid"], ["Circle", "Non-Rigid"],
        ] # NOTE, label_tree_edges format could be changed.
        empty_multiclass_head_indices: []

    All of the member variables should be considered for the Model API.
    https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/algorithms/classification/utils/cls_utils.py#L97
    """

    num_multiclass_heads: int
    num_multilabel_classes: int
    head_idx_to_logits_range: dict[str, tuple[int, int]]
    num_single_label_classes: int
    class_to_group_idx: dict[str, tuple[int, int]]
    all_groups: list[list[str]]
    label_to_idx: dict[str, int]
    label_tree_edges: list[list[str]]
    empty_multiclass_head_indices: list[int]

    @classmethod
    def from_dm_label_groups(cls, dm_label_categories: LabelCategories) -> HLabelInfo:
        """Generate HLabelData from the Datumaro LabelCategories.

        Args:
            dm_label_categories (LabelCategories): the label categories of datumaro.
        """

        def get_exclusive_group_info(all_groups: list[Label | list[Label]]) -> dict[str, Any]:
            """Get exclusive group information."""
            exclusive_groups = [g for g in all_groups if len(g) > 1]

            last_logits_pos = 0
            num_single_label_classes = 0
            head_idx_to_logits_range = {}
            class_to_idx = {}

            for i, group in enumerate(exclusive_groups):
                head_idx_to_logits_range[str(i)] = (last_logits_pos, last_logits_pos + len(group))
                last_logits_pos += len(group)
                for j, c in enumerate(group):
                    class_to_idx[c] = (i, j)
                    num_single_label_classes += 1

            return {
                "num_multiclass_heads": len(exclusive_groups),
                "head_idx_to_logits_range": head_idx_to_logits_range,
                "class_to_idx": class_to_idx,
                "num_single_label_classes": num_single_label_classes,
            }

        def get_single_label_group_info(
            all_groups: list[Label | list[Label]],
            num_exclusive_groups: int,
        ) -> dict[str, Any]:
            """Get single label group information."""
            single_label_groups = [g for g in all_groups if len(g) == 1]

            class_to_idx = {}

            for i, group in enumerate(single_label_groups):
                class_to_idx[group[0]] = (num_exclusive_groups, i)

            return {
                "num_multilabel_classes": len(single_label_groups),
                "class_to_idx": class_to_idx,
            }

        def merge_class_to_idx(
            exclusive_ctoi: dict[str, tuple[int, int]],
            single_label_ctoi: dict[str, tuple[int, int]],
        ) -> dict[str, tuple[int, int]]:
            """Merge the class_to_idx information from exclusive and single_label groups."""

            def put_key_values(src: dict, dst: dict) -> None:
                """Put key and values from src to dst."""
                for k, v in src.items():
                    dst[k] = v

            class_to_idx: dict[str, tuple[int, int]] = {}
            put_key_values(exclusive_ctoi, class_to_idx)
            put_key_values(single_label_ctoi, class_to_idx)
            return class_to_idx

        def get_label_tree_edges(dm_label_items: list[LabelCategories]) -> list[list[str]]:
            """Get label tree edges information. Each edges represent [child, parent]."""
            return [[item.name, item.parent] for item in dm_label_items if item.parent != ""]

        all_groups = [label_group.labels for label_group in dm_label_categories.label_groups]

        exclusive_group_info = get_exclusive_group_info(all_groups)
        single_label_group_info = get_single_label_group_info(all_groups, exclusive_group_info["num_multiclass_heads"])

        merged_class_to_idx = merge_class_to_idx(
            exclusive_group_info["class_to_idx"],
            single_label_group_info["class_to_idx"],
        )

        return HLabelInfo(
            label_names=[item.name for item in dm_label_categories.items],
            label_groups=all_groups,
            num_multiclass_heads=exclusive_group_info["num_multiclass_heads"],
            num_multilabel_classes=single_label_group_info["num_multilabel_classes"],
            head_idx_to_logits_range=exclusive_group_info["head_idx_to_logits_range"],
            num_single_label_classes=exclusive_group_info["num_single_label_classes"],
            class_to_group_idx=merged_class_to_idx,
            all_groups=all_groups,
            label_to_idx=dm_label_categories._indices,  # noqa: SLF001
            label_tree_edges=get_label_tree_edges(dm_label_categories.items),
            empty_multiclass_head_indices=[],  # consider the label removing case
        )

    def as_head_config_dict(self) -> dict[str, Any]:
        """Return a dictionary including params needed to configure the HLabel MMPretrained head network."""
        return {
            "num_classes": self.num_classes,
            "num_multiclass_heads": self.num_multiclass_heads,
            "num_multilabel_classes": self.num_multilabel_classes,
            "head_idx_to_logits_range": self.head_idx_to_logits_range,
            "num_single_label_classes": self.num_single_label_classes,
            "empty_multiclass_head_indices": self.empty_multiclass_head_indices,
        }

    @classmethod
    def from_json(cls, serialized: str) -> HLabelInfo:
        """Reconstruct it from the JSON serialized string."""
        loaded = json.loads(serialized)
        # List to tuple
        loaded["head_idx_to_logits_range"] = {
            key: tuple(value) for key, value in loaded["head_idx_to_logits_range"].items()
        }
        loaded["class_to_group_idx"] = {key: tuple(value) for key, value in loaded["class_to_group_idx"].items()}
        return cls(**loaded)


@dataclass
class SegLabelInfo(LabelInfo):
    """Meta information of Semantic Segmentation."""

    def __init__(self, label_names: list[str], label_groups: list[list[str]], ignore_index: int | None = 255) -> None:
        if not any(word.lower() == "background" for word in label_names):
            msg = (
                "Currently, no background label exists for `label_names`. "
                "Segmentation requires a background label. "
                "To do this, `Background` is added at index 0 of `label_names`."
            )
            warnings.warn(msg, stacklevel=2)
            label_names.insert(0, "Background")
        super().__init__(label_names, label_groups)
        self.ignore_index = ignore_index


@dataclass
class NullLabelInfo(LabelInfo):
    """Represent no label information. It is used for Visual Prompting tasks."""

    def __init__(self) -> None:
        super().__init__(label_names=[], label_groups=[[]])

    @classmethod
    def from_json(cls, _: str) -> LabelInfo:
        """Reconstruct it from the JSON serialized string."""
        return cls()
