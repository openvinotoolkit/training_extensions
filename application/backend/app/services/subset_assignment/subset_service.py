# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from uuid import UUID

from loguru import logger

from app.models import DatasetItemSubset
from app.repositories import DatasetItemRepository, DatasetViewRepository
from app.services import BaseSessionManagedService

from .models import DatasetItemWithLabels, SubsetAssignment


class SubsetService(BaseSessionManagedService):
    def get_unassigned_items_with_labels(
        self, project_id: UUID, dataset_view_id: UUID | None = None
    ) -> list[DatasetItemWithLabels]:
        """
        Retrieve all unassigned dataset items for a given project, optionally restricted to a dataset view.

        Video frames carry their parent video id as ``group_id`` so that the subset assigner can keep frames
        of the same video within a single subset (or warn when that is not possible).
        """
        if dataset_view_id is not None:
            view_repo = DatasetViewRepository(project_id=str(project_id), db=self.db_session)
            unassigned_items_db = view_repo.list_unassigned_items(str(dataset_view_id))
        else:
            repo = DatasetItemRepository(project_id=str(project_id), db=self.db_session)
            unassigned_items_db = repo.list_unassigned_items()

        items_dict = defaultdict(set)
        group_by_item: dict[str, UUID | None] = {}
        for label, video_id in unassigned_items_db:
            items_dict[label.dataset_item_id].add(UUID(label.label_id))
            group_by_item[label.dataset_item_id] = UUID(video_id) if video_id else None

        return [
            DatasetItemWithLabels(item_id=UUID(item_id), labels=labels, group_id=group_by_item[item_id])
            for item_id, labels in items_dict.items()
        ]

    def get_pinned_group_subsets(self, project_id: UUID) -> dict[UUID, DatasetItemSubset]:
        """Map each video that already has assigned frames to its established subset.

        Newly annotated frames of such a video must join that subset, otherwise
        incremental training runs would split near-duplicate frames across subsets.
        If a video's frames are (historically) spread over several subsets, the subset
        holding most of its frames wins; ties break deterministically by subset order.
        """
        repo = DatasetItemRepository(project_id=str(project_id), db=self.db_session)
        counts: dict[UUID, dict[DatasetItemSubset, int]] = defaultdict(dict)
        for video_id, subset, count in repo.list_assigned_group_subsets():
            counts[UUID(video_id)][DatasetItemSubset(subset)] = count

        subset_order = list(DatasetItemSubset)
        return {
            video_id: max(by_subset, key=lambda s: (by_subset[s], -subset_order.index(s)))
            for video_id, by_subset in counts.items()
        }

    def has_all_subsets_assigned(self, project_id: UUID, dataset_view_id: UUID | None = None) -> bool:
        """Return True if there is at least one dataset item for each of TRAINING, VALIDATION, and TESTING subsets."""
        if dataset_view_id is not None:
            view_repo = DatasetViewRepository(project_id=str(project_id), db=self.db_session)
            return view_repo.has_all_subsets_assigned(str(dataset_view_id))
        repo = DatasetItemRepository(project_id=str(project_id), db=self.db_session)
        return repo.has_all_subsets_assigned()

    def update_subset_assignments(self, project_id: UUID, assignments: list[SubsetAssignment]) -> None:
        """Update subset assignments for dataset items."""
        repo = DatasetItemRepository(project_id=str(project_id), db=self.db_session)

        assignments_by_subset = defaultdict(set)
        for assignment in assignments:
            assignments_by_subset[assignment.subset].add(str(assignment.item_id))

        for subset, item_ids in assignments_by_subset.items():
            logger.info("Updating subset assignments for {}: {} items", subset, len(item_ids))
            repo.set_subset(obj_ids=item_ids, subset=subset)
