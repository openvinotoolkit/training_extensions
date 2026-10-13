# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from uuid import UUID

import numpy as np
from loguru import logger

from app.models import DatasetItemSubset

from .models import DatasetItemWithLabels, SplitRatios, SubsetAssignment

# Minimum number of distinct media groups required for group-aware assignment:
# with fewer groups than subsets, groups cannot be distributed without leakage.
MIN_GROUPS_FOR_GROUP_AWARE_SPLIT = 3


class SubsetAssigner:
    """
    Assigns dataset items to subsets ensuring balanced label representation.

    Uses iterative stratification to handle both single-label and multi-label classification problems.
    Items that share a media group (e.g. frames of the same video, identified by
    ``DatasetItemWithLabels.group_id``) are near-duplicates for evaluation purposes: whenever enough
    distinct groups exist, whole groups are assigned to a single subset so that no video leaks
    across the training/validation/testing boundary. When grouping is impossible (e.g. all
    annotated frames come from one video), the assigner falls back to per-item assignment and logs
    a warning that evaluation metrics may overestimate real-world performance.

    Sources:
    Sechidis, K., Tsoumakas, G., & Vlahavas, I. (2011). On the stratification of multi-label data.
    Machine Learning and Knowledge Discovery in Databases, 145-158.
    http://lpis.csd.auth.gr/publications/sechidis-ecmlpkdd-2011.pdf

    Piotr Szymański, Tomasz Kajdanowicz ; Proceedings of the First International Workshop on Learning
    with Imbalanced Domains: Theory and Applications, PMLR 74:22-35, 2017.
    http://proceedings.mlr.press/v74/szyma%C5%84ski17a.html
    """

    def __init__(self) -> None:
        from sklearn.preprocessing import MultiLabelBinarizer

        self._mlb = MultiLabelBinarizer()

    def assign(
        self,
        items: list[DatasetItemWithLabels],
        target_ratios: SplitRatios,
        has_all_subsets_assigned: bool,
        pinned_group_subsets: dict[UUID, DatasetItemSubset] | None = None,
    ) -> list[SubsetAssignment]:
        """
        Assigns dataset items to subsets based on target ratios.

        Args:
            items (list[DatasetItemWithLabels]): List of dataset items to assign.
            target_ratios (SplitRatios): Desired split ratios for subsets.
            has_all_subsets_assigned (bool): Whether the project already has at least one item
                assigned to each of the TRAINING, VALIDATION, and TESTING subsets.
                - If ``False`` and fewer than 3 items are provided, a ``ValueError`` is raised
                  because there are not enough items to populate every subset.  After stratification,
                  ``_ensure_all_subsets_nonempty`` is called to guarantee no subset is left empty.
                - If ``True``, the minimum-item guard is bypassed (the caller guarantees that each
                  subset is already covered).  When fewer than 3 items are provided, their media
                  groups are assigned sequentially (TRAINING first, then VALIDATION, then TESTING).
                  ``_ensure_all_subsets_nonempty`` is *not* called, so stratification may produce
                  empty folds without raising.
            pinned_group_subsets (dict[UUID, DatasetItemSubset] | None): Media groups that already
                have assigned items in the project, mapped to their established subset. Items of
                these groups always join the pinned subset, so incremental annotation of a video
                cannot spread its frames across subsets.
        Returns:
            list[SubsetAssignment]: List of subset assignments for each item.
        """
        pinned_assignments, items = self._apply_pins(items, pinned_group_subsets or {})

        if len(items) < 3:
            if not has_all_subsets_assigned:
                raise ValueError(
                    "Not all subsets have items assigned, but number of unassigned dataset items is less than number "
                    "of subsets: Training, Validation and Testing. Each subset requires at least 1 item before "
                    "training can start."
                )
            # Fewer items than subsets: stratification is impossible, so assign media groups
            # (not individual items) to subsets in order, keeping each group in one subset.
            assignments = list(pinned_assignments)
            subsets = [DatasetItemSubset.TRAINING, DatasetItemSubset.VALIDATION, DatasetItemSubset.TESTING]
            for pos, indices in enumerate(self._group_items(items).values()):
                for idx in indices:
                    assignments.append(SubsetAssignment(item_id=items[idx].item_id, subset=subsets[pos]))
            return assignments

        items_by_group = self._group_items(items)
        group_keys = list(items_by_group.keys())
        has_multi_item_groups = any(len(indices) > 1 for indices in items_by_group.values())
        use_groups = has_multi_item_groups and len(group_keys) >= MIN_GROUPS_FOR_GROUP_AWARE_SPLIT

        if use_groups:
            # Stratify at the group level so that no media group spans multiple subsets.
            labels_per_unit = [frozenset().union(*(items[i].labels for i in items_by_group[key])) for key in group_keys]
        else:
            self._warn_on_unavoidable_leakage(items, items_by_group, has_multi_item_groups)
            labels_per_unit = [item.labels for item in items]

        indices_by_subset = self._stratify(labels_per_unit, target_ratios)

        if not has_all_subsets_assigned:
            self._ensure_all_subsets_nonempty(indices_by_subset)

        if use_groups:
            return pinned_assignments + self._group_assignments(items, indices_by_subset, items_by_group, group_keys)
        return pinned_assignments + [
            SubsetAssignment(item_id=items[idx].item_id, subset=subset)
            for subset, indices in indices_by_subset.items()
            for idx in indices
        ]

    @staticmethod
    def _apply_pins(
        items: list[DatasetItemWithLabels], pins: dict[UUID, DatasetItemSubset]
    ) -> tuple[list[SubsetAssignment], list[DatasetItemWithLabels]]:
        """Assign items of already-established media groups to their pinned subset.

        Returns the resulting assignments and the remaining (free) items."""
        if not pins:
            return [], items
        pinned_assignments: list[SubsetAssignment] = []
        free_items: list[DatasetItemWithLabels] = []
        for item in items:
            subset = pins.get(item.group_id) if item.group_id else None
            if subset is None:
                free_items.append(item)
            else:
                pinned_assignments.append(SubsetAssignment(item_id=item.item_id, subset=subset))
        return pinned_assignments, free_items

    def _stratify(self, labels_per_unit: list, target_ratios: SplitRatios) -> dict[DatasetItemSubset, list[int]]:
        """Run iterative stratification over the given units (items or media groups)."""
        from skmultilearn.model_selection import IterativeStratification

        label_matrix = self._mlb.fit_transform(labels_per_unit)

        stratifier = IterativeStratification(
            n_splits=3,
            order=1,
            sample_distribution_per_fold=target_ratios.to_list(),
        )

        X = np.arange(len(labels_per_unit)).reshape(-1, 1)
        y = label_matrix

        splits = list(stratifier.split(X, y))  # pyrefly: ignore[bad-argument-type]
        return {
            DatasetItemSubset.TRAINING: splits[0][1].tolist(),
            DatasetItemSubset.VALIDATION: splits[1][1].tolist(),
            DatasetItemSubset.TESTING: splits[2][1].tolist(),
        }

    @staticmethod
    def _group_items(items: list[DatasetItemWithLabels]) -> dict[object, list[int]]:
        """Group item indices by media group; items without a group form singleton groups."""
        items_by_group: dict[object, list[int]] = defaultdict(list)
        for idx, item in enumerate(items):
            items_by_group[item.group_id or item.item_id].append(idx)
        return items_by_group

    @staticmethod
    def _group_assignments(
        items: list[DatasetItemWithLabels],
        indices_by_subset: dict[DatasetItemSubset, list[int]],
        items_by_group: dict[object, list[int]],
        group_keys: list,
    ) -> list[SubsetAssignment]:
        """Expand group-level subset indices into per-item assignments."""
        return [
            SubsetAssignment(item_id=items[idx].item_id, subset=subset)
            for subset, group_indices in indices_by_subset.items()
            for gidx in group_indices
            for idx in items_by_group[group_keys[gidx]]
        ]

    @staticmethod
    def _warn_on_unavoidable_leakage(
        items: list[DatasetItemWithLabels],
        items_by_group: dict[object, list[int]],
        has_multi_item_groups: bool,
    ) -> None:
        """Warn when items sharing a media group are about to be split across subsets."""
        if not has_multi_item_groups:
            return
        logger.warning(
            "All {} annotated items belong to only {} media group(s) (e.g. frames of the same "
            "video). Training, validation and testing subsets will contain near-duplicate items, "
            "so evaluation metrics may substantially overestimate real-world performance. "
            "Annotate frames from additional videos or standalone images for a reliable evaluation.",
            len(items),
            len(items_by_group),
        )

    @staticmethod
    def _ensure_all_subsets_nonempty(
        indices_by_subset: dict[DatasetItemSubset, list[int]],
    ) -> None:
        """Ensure every subset has at least one index by moving items from the largest subset."""
        empty_subsets = [s for s, idx in indices_by_subset.items() if len(idx) == 0]
        for empty_subset in empty_subsets:
            largest_subset = max(indices_by_subset, key=lambda s: len(indices_by_subset[s]))
            moved = indices_by_subset[largest_subset].pop()
            indices_by_subset[empty_subset].append(moved)
