# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

import pytest

from app.models import DatasetItemSubset
from app.services.subset_assignment import DatasetItemWithLabels, SplitRatios, SubsetAssigner


class TestSubsetAssigner:
    """Test cases for SubsetAssigner class"""

    @pytest.fixture
    def fxt_assigner(self):
        """Create a SubsetAssigner instance"""
        return SubsetAssigner()

    @pytest.fixture
    def fxt_default_ratios(self):
        """Create default split ratios"""
        return SplitRatios(train=0.7, val=0.2, test=0.1)

    def test_assign_empty_list(self, fxt_assigner, fxt_default_ratios):
        """Test that assigning empty list returns empty list"""
        result = fxt_assigner.assign([], fxt_default_ratios, has_all_subsets_assigned=True)
        assert result == []

    def test_assign_returns_all_items(self, fxt_assigner, fxt_default_ratios):
        """Test that all input items are assigned to a subset"""
        items = [DatasetItemWithLabels(item_id=uuid4(), labels={uuid4()}) for _ in range(100)]

        result = fxt_assigner.assign(items, fxt_default_ratios, has_all_subsets_assigned=False)

        assert len(result) == len(items)

        # Check all item IDs are present
        result_item_ids = {assignment.item_id for assignment in result}
        input_item_ids = {item.item_id for item in items}
        assert result_item_ids == input_item_ids

    def test_assign_maintains_label_distribution(self, fxt_assigner, fxt_default_ratios):
        """Test that label distribution is maintained across subsets (stratification)"""
        label_a = uuid4()
        label_b = uuid4()

        items = []
        # 60% label A, 40% label B
        for _ in range(60):
            items.append(DatasetItemWithLabels(item_id=uuid4(), labels={label_a}))
        for _ in range(40):
            items.append(DatasetItemWithLabels(item_id=uuid4(), labels={label_b}))

        result = fxt_assigner.assign(items, fxt_default_ratios, has_all_subsets_assigned=False)

        item_labels_map = {item.item_id: item.labels for item in items}

        # Check label distribution in each subset
        for subset in [DatasetItemSubset.TRAINING, DatasetItemSubset.VALIDATION, DatasetItemSubset.TESTING]:
            subset_assignments = [a for a in result if a.subset == subset]

            label_a_count = sum(1 for a in subset_assignments if label_a in item_labels_map[a.item_id])
            label_b_count = sum(1 for a in subset_assignments if label_b in item_labels_map[a.item_id])

            total_in_subset = len(subset_assignments)
            if total_in_subset > 0:
                # Label A should be ~60% and label B ~40% in each subset
                assert label_a_count / total_in_subset == 0.6
                assert label_b_count / total_in_subset == 0.4

    def test_assign_raises_when_too_few_items_and_not_all_subsets_assigned(self, fxt_assigner, fxt_default_ratios):
        """Test that a ValueError is raised when fewer than 3 items exist and not all subsets are already assigned."""
        items = [
            DatasetItemWithLabels(item_id=uuid4(), labels={uuid4()}),
            DatasetItemWithLabels(item_id=uuid4(), labels={uuid4()}),
        ]

        with pytest.raises(ValueError, match="number of unassigned dataset items is less than number of subsets"):
            fxt_assigner.assign(items, fxt_default_ratios, has_all_subsets_assigned=False)

    def test_assign_allows_fewer_than_three_items_when_all_subsets_already_assigned(
        self, fxt_assigner, fxt_default_ratios
    ):
        """When has_all_subsets_assigned=True, the minimum-unassigned-items guard is bypassed.

        When has_all_subsets_assigned=True, the guard is bypassed and the call succeeds even with
        fewer items than subsets.
        """
        items = [DatasetItemWithLabels(item_id=uuid4(), labels={uuid4()}) for _ in range(2)]

        result = fxt_assigner.assign(items, fxt_default_ratios, has_all_subsets_assigned=True)
        assert len(result) == len(items)

    def test_assign_ensures_all_subsets_nonempty_when_not_all_subsets_assigned(self, fxt_assigner, fxt_default_ratios):
        """Test that every subset receives at least one item when not all subsets are pre-assigned.

        With has_all_subsets_assigned=False and enough items, _ensure_all_subsets_nonempty
        must guarantee each of TRAINING, VALIDATION, and TESTING has at least one assignment.
        """
        # Use exactly 3 items - minimum allowed when has_all_subsets_assigned=False
        items = [DatasetItemWithLabels(item_id=uuid4(), labels={uuid4()}) for _ in range(3)]

        result = fxt_assigner.assign(items, fxt_default_ratios, has_all_subsets_assigned=False)

        assigned_subsets = {assignment.subset for assignment in result}
        assert DatasetItemSubset.TRAINING in assigned_subsets
        assert DatasetItemSubset.VALIDATION in assigned_subsets
        assert DatasetItemSubset.TESTING in assigned_subsets

    def test_assign_does_not_enforce_nonempty_subsets_when_all_subsets_already_assigned(
        self, fxt_assigner, fxt_default_ratios
    ):
        """Test that _ensure_all_subsets_nonempty is NOT called when all subsets are pre-assigned.

        When has_all_subsets_assigned=True the assigner trusts that the existing data already
        covers all three subsets, so it does not forcibly move items between subsets.
        Use a pathological ratio (1/0/0) to produce an empty val/test fold and verify no
        redistribution occurs.
        """
        skewed_ratios = SplitRatios(train=0.98, val=0.01, test=0.01)
        # 3 items with a heavily skewed ratio - stratifier will put all 3 in train, leaving
        # val and test empty.  With has_all_subsets_assigned=True this must NOT raise.
        items = [DatasetItemWithLabels(item_id=uuid4(), labels={uuid4()}) for _ in range(3)]

        result = fxt_assigner.assign(items, skewed_ratios, has_all_subsets_assigned=True)

        # All items are assigned (no items lost)
        assert len(result) == len(items)
        # With no redistribution, all items stay in TRAINING; val and test remain empty
        assigned_subsets = {assignment.subset for assignment in result}
        assert DatasetItemSubset.TRAINING in assigned_subsets
        assert DatasetItemSubset.VALIDATION not in assigned_subsets
        assert DatasetItemSubset.TESTING not in assigned_subsets

    @pytest.mark.parametrize(
        "num_items, expected_subsets",
        [
            (1, [DatasetItemSubset.TRAINING]),
            (2, [DatasetItemSubset.TRAINING, DatasetItemSubset.VALIDATION]),
        ],
    )
    def test_assign_fewer_than_three_items_assigns_in_subset_order(
        self, fxt_assigner, fxt_default_ratios, num_items, expected_subsets
    ):
        """Test that when fewer than 3 items are provided and all subsets are already assigned,
        items are assigned in order: TRAINING, VALIDATION, TESTING."""
        items = [DatasetItemWithLabels(item_id=uuid4(), labels={uuid4()}) for _ in range(num_items)]

        result = fxt_assigner.assign(items, fxt_default_ratios, has_all_subsets_assigned=True)

        assert len(result) == num_items
        # Verify items are assigned to subsets in the expected order
        for assignment, expected_subset in zip(result, expected_subsets):
            assert assignment.subset == expected_subset
        # Verify all input item IDs are present in the result
        result_item_ids = {a.item_id for a in result}
        input_item_ids = {item.item_id for item in items}
        assert result_item_ids == input_item_ids


class TestSubsetAssignerMediaGroups:
    """Group-aware assignment: items sharing a media group (e.g. frames of the same
    video) must not be split across subsets, and impossible cases must not crash."""

    @pytest.fixture
    def fxt_assigner(self):
        return SubsetAssigner()

    @pytest.fixture
    def fxt_default_ratios(self):
        return SplitRatios(train=0.7, val=0.2, test=0.1)

    def test_grouped_items_stay_in_same_subset(self, fxt_assigner, fxt_default_ratios):
        """Frames of the same video must all land in one subset."""
        label = uuid4()
        groups = [uuid4() for _ in range(10)]
        items = [
            DatasetItemWithLabels(item_id=uuid4(), labels={label}, group_id=group)
            for group in groups
            for _ in range(12)
        ]

        result = fxt_assigner.assign(items, fxt_default_ratios, has_all_subsets_assigned=False)

        assert len(result) == len(items)
        group_of_item = {item.item_id: item.group_id for item in items}
        subsets_per_group = {}
        for assignment in result:
            subsets_per_group.setdefault(group_of_item[assignment.item_id], set()).add(assignment.subset)
        for group, subsets in subsets_per_group.items():
            assert len(subsets) == 1, f"group {group} was split across subsets {subsets}"
        # every subset must be populated
        assert {a.subset for a in result} == {
            DatasetItemSubset.TRAINING,
            DatasetItemSubset.VALIDATION,
            DatasetItemSubset.TESTING,
        }

    def test_single_group_falls_back_and_warns(self, fxt_assigner, fxt_default_ratios, monkeypatch):
        """With all frames from one video, grouping is impossible: assignment must
        still succeed (frame-level split) and a warning must be emitted."""
        from app.services.subset_assignment import assigner as assigner_module

        warnings = []
        monkeypatch.setattr(assigner_module.logger, "warning", lambda msg, *args, **kwargs: warnings.append(msg))

        video = uuid4()
        label = uuid4()
        items = [DatasetItemWithLabels(item_id=uuid4(), labels={label}, group_id=video) for _ in range(30)]

        result = fxt_assigner.assign(items, fxt_default_ratios, has_all_subsets_assigned=False)

        assert len(result) == len(items)
        assert {a.subset for a in result} == {
            DatasetItemSubset.TRAINING,
            DatasetItemSubset.VALIDATION,
            DatasetItemSubset.TESTING,
        }
        assert warnings, "expected a leakage warning when all items share one media group"

    def test_items_without_group_behave_as_before(self, fxt_assigner, fxt_default_ratios):
        """Images (group_id=None) keep the original per-item stratification."""
        label = uuid4()
        items = [DatasetItemWithLabels(item_id=uuid4(), labels={label}) for _ in range(50)]

        result = fxt_assigner.assign(items, fxt_default_ratios, has_all_subsets_assigned=False)

        assert len(result) == len(items)
        assert {a.subset for a in result} == {
            DatasetItemSubset.TRAINING,
            DatasetItemSubset.VALIDATION,
            DatasetItemSubset.TESTING,
        }

    def test_mixed_groups_and_images(self, fxt_assigner, fxt_default_ratios):
        """Videos and standalone images can be mixed; groups stay atomic."""
        label = uuid4()
        video_a, video_b, video_c = uuid4(), uuid4(), uuid4()
        items = (
            [DatasetItemWithLabels(item_id=uuid4(), labels={label}, group_id=video_a) for _ in range(10)]
            + [DatasetItemWithLabels(item_id=uuid4(), labels={label}, group_id=video_b) for _ in range(10)]
            + [DatasetItemWithLabels(item_id=uuid4(), labels={label}, group_id=video_c) for _ in range(10)]
            + [DatasetItemWithLabels(item_id=uuid4(), labels={label}) for _ in range(15)]
        )

        result = fxt_assigner.assign(items, fxt_default_ratios, has_all_subsets_assigned=False)

        assert len(result) == len(items)
        group_of_item = {item.item_id: item.group_id for item in items}
        for video in (video_a, video_b, video_c):
            subsets = {a.subset for a in result if group_of_item[a.item_id] == video}
            assert len(subsets) == 1, f"video group split across {subsets}"


class TestSubsetAssignerPinnedGroups:
    """Incremental annotation: frames of a video that already has assigned frames
    must join that video's established subset instead of being re-stratified."""

    @pytest.fixture
    def fxt_assigner(self):
        return SubsetAssigner()

    @pytest.fixture
    def fxt_default_ratios(self):
        return SplitRatios(train=0.7, val=0.2, test=0.1)

    def test_pinned_group_items_join_established_subset(self, fxt_assigner, fxt_default_ratios):
        label = uuid4()
        pinned_video = uuid4()
        free_videos = [uuid4() for _ in range(5)]
        pinned_items = [DatasetItemWithLabels(item_id=uuid4(), labels={label}, group_id=pinned_video) for _ in range(7)]
        free_items = [
            DatasetItemWithLabels(item_id=uuid4(), labels={label}, group_id=video)
            for video in free_videos
            for _ in range(10)
        ]

        result = fxt_assigner.assign(
            pinned_items + free_items,
            fxt_default_ratios,
            has_all_subsets_assigned=True,
            pinned_group_subsets={pinned_video: DatasetItemSubset.TESTING},
        )

        assert len(result) == len(pinned_items) + len(free_items)
        pinned_ids = {item.item_id for item in pinned_items}
        for assignment in result:
            if assignment.item_id in pinned_ids:
                assert assignment.subset == DatasetItemSubset.TESTING

    def test_small_incremental_batch_keeps_video_frames_together(self, fxt_assigner, fxt_default_ratios):
        """Two new frames of the same video (fewer items than subsets) must land in ONE subset."""
        label = uuid4()
        video = uuid4()
        items = [DatasetItemWithLabels(item_id=uuid4(), labels={label}, group_id=video) for _ in range(2)]

        result = fxt_assigner.assign(items, fxt_default_ratios, has_all_subsets_assigned=True)

        assert len(result) == 2
        assert len({a.subset for a in result}) == 1, "frames of one video were split across subsets"

    def test_small_batch_of_independent_items_keeps_sequential_behavior(self, fxt_assigner, fxt_default_ratios):
        """Two unrelated items (fewer than subsets) keep the original sequential assignment."""
        label = uuid4()
        items = [DatasetItemWithLabels(item_id=uuid4(), labels={label}) for _ in range(2)]

        result = fxt_assigner.assign(items, fxt_default_ratios, has_all_subsets_assigned=True)

        assert [a.subset for a in result] == [DatasetItemSubset.TRAINING, DatasetItemSubset.VALIDATION]

    def test_small_incremental_batch_respects_pins(self, fxt_assigner, fxt_default_ratios):
        """New frames of an established video are pinned even below the size guard."""
        label = uuid4()
        video = uuid4()
        items = [DatasetItemWithLabels(item_id=uuid4(), labels={label}, group_id=video) for _ in range(2)]

        result = fxt_assigner.assign(
            items,
            fxt_default_ratios,
            has_all_subsets_assigned=True,
            pinned_group_subsets={video: DatasetItemSubset.VALIDATION},
        )

        assert len(result) == 2
        assert all(a.subset == DatasetItemSubset.VALIDATION for a in result)
