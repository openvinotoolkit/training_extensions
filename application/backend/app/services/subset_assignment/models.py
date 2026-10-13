# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from uuid import UUID

from app.models import DatasetItemSubset


@dataclass(frozen=True)
class SplitRatios:
    """Value object representing split ratios for subset assignment"""

    train: float
    val: float
    test: float

    def __post_init__(self):
        total = self.train + self.val + self.test
        if not (0.99 <= total <= 1.01):  # Allow for floating point precision issues
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    def to_fold_sizes(self, total_items: int) -> dict[DatasetItemSubset, int]:
        """
        Convert split ratios to absolute fold sizes based on total items

        Each fold is guaranteed to receive at least 1 item. Fold sizes are calculated by rounding proportional
        allocations. If the sum differs from total_items due to rounding, the difference is adjusted by adding/removing
        items from folds in order of their ratio size (largest first).
        When ratios are equal, priority is: val > train > test.

        Args:
            total_items: Total number of items to distribute across folds

        Returns:
            Dictionary mapping each subset to its absolute size

        Raises:
            ValueError: If total_items < 3 (need at least 1 item per fold)
        """
        if total_items < 3:
            raise ValueError(f"Need at least 3 items to create all subsets, got {total_items}")

        # Ensure minimum of 1 per fold
        train_size = max(1, round(total_items * self.train))
        val_size = max(1, round(total_items * self.val))
        test_size = max(1, round(total_items * self.test))

        # Adjust to match exact total by reducing from the largest fold
        diff = (train_size + val_size + test_size) - total_items
        if diff != 0:
            # Sort by ratio to adjust the largest first, prioritizing val over the train if equal
            folds = [(self.val, "val"), (self.train, "train"), (self.test, "test")]
            folds.sort(reverse=True, key=lambda x: x[0])

            sizes = {"train": train_size, "val": val_size, "test": test_size}

            for _, fold_name in folds:
                if diff > 0:  # Over-allocated: reduce
                    if sizes[fold_name] > 1:
                        reduction = min(sizes[fold_name] - 1, diff)
                        sizes[fold_name] -= reduction
                        diff -= reduction
                elif diff < 0:  # Under-allocated: increase
                    addition = abs(diff)
                    sizes[fold_name] += addition
                    diff += addition
                if diff == 0:
                    break

            train_size, val_size, test_size = sizes["train"], sizes["val"], sizes["test"]

        return {
            DatasetItemSubset.TRAINING: train_size,
            DatasetItemSubset.VALIDATION: val_size,
            DatasetItemSubset.TESTING: test_size,
        }

    def to_list(self) -> list[float]:
        """Convert split ratios to a list"""
        return [self.train, self.val, self.test]


@dataclass(frozen=True)
class SubsetAssignment:
    """Value object representing assignment of a dataset item to a subset"""

    item_id: UUID
    subset: DatasetItemSubset


@dataclass(frozen=True)
class DatasetItemWithLabels:
    """Representation of a dataset item with its associated labels for assignment processing.

    ``group_id`` identifies the media group the item belongs to (e.g. the parent video
    of an annotated video frame). Items sharing a group are near-duplicates for
    evaluation purposes and should not be split across subsets. ``None`` means the
    item is its own group (e.g. an image).
    """

    item_id: UUID
    labels: set[UUID]
    group_id: UUID | None = None
