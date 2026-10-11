# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.schema import DatasetItemDB, DatasetViewDB, DatasetViewItemDB, LabelDB, ProjectDB
from app.models import DatasetItemSubset, TaskType
from app.services.subset_assignment import SubsetAssignment, SubsetService
from tests.integration.project_factory import ProjectTestDataFactory


@pytest.fixture(autouse=True)
def setup_project_with_dataset_items(
    fxt_db_projects: list[ProjectDB],
    fxt_db_labels: list[LabelDB],
    fxt_default_distribution: dict[DatasetItemSubset, int],
    db_session: Session,
) -> None:
    """Fixture to set up a project with dataset items in the database."""

    db_label = fxt_db_labels[0]
    (
        ProjectTestDataFactory(db_session)
        .with_project(fxt_db_projects[0])
        .with_label(db_label)
        .with_media_and_dataset_items(fxt_default_distribution)
        .with_item_labels(db_label)
        .build()
    )


@pytest.fixture
def fxt_subset_service(db_session: Session) -> SubsetService:
    """Fixture to create a SubsetService instance."""
    return SubsetService(db_session)


@pytest.fixture
def fxt_default_distribution() -> dict[DatasetItemSubset, int]:
    """Fixture to provide a default subset distribution."""
    return {
        DatasetItemSubset.TRAINING: 70,
        DatasetItemSubset.VALIDATION: 20,
        DatasetItemSubset.TESTING: 10,
        DatasetItemSubset.UNASSIGNED: 50,
    }


class TestSubsetServiceIntegration:
    """Integration tests for SubsetService."""

    def test_get_unassigned_items_with_labels(
        self,
        fxt_project_id: UUID,
        fxt_subset_service: SubsetService,
        fxt_default_distribution: dict[DatasetItemSubset, int],
    ) -> None:
        """Test retrieving unassigned dataset items with labels."""
        items = fxt_subset_service.get_unassigned_items_with_labels(fxt_project_id)
        assert len(items) == fxt_default_distribution.get(DatasetItemSubset.UNASSIGNED)  # based on default distribution

    def test_update_subset_assignments(
        self,
        fxt_project_id: UUID,
        fxt_subset_service: SubsetService,
        fxt_default_distribution: dict[DatasetItemSubset, int],
        db_session: Session,
    ):
        """Test updating subset assignments."""
        items = fxt_subset_service.get_unassigned_items_with_labels(fxt_project_id)
        assignments: list[SubsetAssignment] = [
            *[SubsetAssignment(item_id=item.item_id, subset=DatasetItemSubset.TRAINING) for item in items[:30]],
            *[SubsetAssignment(item_id=item.item_id, subset=DatasetItemSubset.VALIDATION) for item in items[30:40]],
            *[SubsetAssignment(item_id=item.item_id, subset=DatasetItemSubset.TESTING) for item in items[40:]],
        ]

        fxt_subset_service.update_subset_assignments(fxt_project_id, assignments)

        # Verify that there are no unassigned items left
        assert (
            db_session.scalar(
                select(func.count())
                .select_from(DatasetItemDB)
                .where(DatasetItemDB.subset == DatasetItemSubset.UNASSIGNED)
            )
            == 0
        )

        # Verify that items have been assigned to corresponding subset using the current distribution values and
        # the new assignments
        distribute_counts = {
            DatasetItemSubset.TRAINING: 30,
            DatasetItemSubset.VALIDATION: 10,
            DatasetItemSubset.TESTING: 10,
        }
        for subset, count in distribute_counts.items():
            assert (
                db_session.scalar(select(func.count()).select_from(DatasetItemDB).where(DatasetItemDB.subset == subset))
                == fxt_default_distribution.get(subset, 0) + count
            )

    def test_has_all_subsets_assigned_returns_true_when_all_three_subsets_present(
        self,
        fxt_project_id: UUID,
        fxt_subset_service: SubsetService,
        fxt_default_distribution: dict[DatasetItemSubset, int],
    ) -> None:
        """Test that has_all_subsets_assigned returns True when TRAINING, VALIDATION, and TESTING each
        have at least one item - which is the case with the default fixture distribution."""
        # The default distribution already contains TRAINING, VALIDATION, and TESTING items
        for subset in (DatasetItemSubset.TRAINING, DatasetItemSubset.VALIDATION, DatasetItemSubset.TESTING):
            assert fxt_default_distribution.get(subset, 0) > 0, (
                f"Prerequisite: default distribution must have {subset} items"
            )

        assert fxt_subset_service.has_all_subsets_assigned(fxt_project_id) is True

    def test_has_all_subsets_assigned_returns_false_when_a_subset_is_missing(
        self,
        fxt_db_projects: list[ProjectDB],
        db_session: Session,
        fxt_subset_service: SubsetService,
    ) -> None:
        """Test that has_all_subsets_assigned returns False when at least one required subset has no items.

          We create a second project that only has TRAINING and VALIDATION items - no TESTING items -
        so has_all_subsets_assigned must return False for it.
        """
        # Build a second project with only TRAINING + VALIDATION items (no TESTING)
        partial_project = ProjectDB(
            id=str(uuid4()),
            name="Partial Subsets Project",
            task_type=TaskType.DETECTION,
            exclusive_labels=False,
        )
        db_session.add(partial_project)
        db_session.flush()

        incomplete_distribution = {
            DatasetItemSubset.TRAINING: 3,
            DatasetItemSubset.VALIDATION: 2,
            # DatasetItemSubset.TESTING intentionally absent
        }

        ProjectTestDataFactory(db_session).with_project(partial_project).with_media_and_dataset_items(
            incomplete_distribution
        ).build()

        partial_project_id = UUID(partial_project.id)
        partial_service = SubsetService(db_session)
        assert partial_service.has_all_subsets_assigned(partial_project_id) is False

    def test_has_all_subsets_assigned_returns_false_for_all_unassigned_project(
        self,
        fxt_db_projects: list[ProjectDB],
        db_session: Session,
    ) -> None:
        """Test that has_all_subsets_assigned returns False when all items are UNASSIGNED."""
        unassigned_project = ProjectDB(
            id=str(uuid4()),
            name="All Unassigned Project",
            task_type=TaskType.DETECTION,
            exclusive_labels=False,
        )
        db_session.add(unassigned_project)
        db_session.flush()

        only_unassigned = {DatasetItemSubset.UNASSIGNED: 5}

        ProjectTestDataFactory(db_session).with_project(unassigned_project).with_media_and_dataset_items(
            only_unassigned
        ).build()

        unassigned_project_id = UUID(unassigned_project.id)
        service = SubsetService(db_session)
        assert service.has_all_subsets_assigned(unassigned_project_id) is False

    def test_has_all_subsets_assigned_returns_false_for_empty_project(
        self,
        db_session: Session,
    ) -> None:
        """Test that has_all_subsets_assigned returns False when the project has no dataset items at all."""
        empty_project = ProjectDB(
            id=str(uuid4()),
            name="Empty Project",
            task_type=TaskType.DETECTION,
            exclusive_labels=False,
        )
        db_session.add(empty_project)
        db_session.flush()

        empty_project_id = UUID(empty_project.id)
        service = SubsetService(db_session)
        assert service.has_all_subsets_assigned(empty_project_id) is False


class TestSubsetServiceDatasetViewScope:
    """Integration tests for the view-scoped variants of the SubsetService queries."""

    @pytest.fixture
    def fxt_dataset_view_with_items(
        self, fxt_project_id: UUID, db_session: Session
    ) -> Callable[[list[DatasetItemSubset]], UUID]:
        """Returns a callable that creates a view containing one dataset item per requested subset."""

        def _create_view(subsets: list[DatasetItemSubset]) -> UUID:
            view = DatasetViewDB(project_id=str(fxt_project_id), name=f"View {uuid4()}")
            db_session.add(view)
            db_session.flush()

            media_ids = []
            for subset in subsets:
                item_id = db_session.scalar(
                    select(DatasetItemDB.id)
                    .where(DatasetItemDB.project_id == str(fxt_project_id), DatasetItemDB.subset == subset)
                    .limit(1)
                )
                assert item_id is not None, f"Prerequisite: the project must have a {subset} item"
                media_ids.append(item_id)

            db_session.add_all(DatasetViewItemDB(dataset_view_id=view.id, media_id=media_id) for media_id in media_ids)
            db_session.flush()
            return UUID(view.id)

        return _create_view

    def test_get_unassigned_items_is_scoped_to_the_view(
        self,
        fxt_project_id: UUID,
        fxt_subset_service: SubsetService,
        fxt_dataset_view_with_items: Callable[[list[DatasetItemSubset]], UUID],
    ) -> None:
        view_id = fxt_dataset_view_with_items([DatasetItemSubset.UNASSIGNED, DatasetItemSubset.TRAINING])

        items = fxt_subset_service.get_unassigned_items_with_labels(fxt_project_id, dataset_view_id=view_id)

        assert len(items) == 1

    def test_has_all_subsets_assigned_is_scoped_to_the_view(
        self,
        fxt_project_id: UUID,
        fxt_subset_service: SubsetService,
        fxt_dataset_view_with_items: Callable[[list[DatasetItemSubset]], UUID],
    ) -> None:
        complete_view = fxt_dataset_view_with_items(
            [DatasetItemSubset.TRAINING, DatasetItemSubset.VALIDATION, DatasetItemSubset.TESTING]
        )
        partial_view = fxt_dataset_view_with_items([DatasetItemSubset.TRAINING, DatasetItemSubset.VALIDATION])

        # The project as a whole has all subsets, but the partial view does not.
        assert fxt_subset_service.has_all_subsets_assigned(fxt_project_id) is True
        assert fxt_subset_service.has_all_subsets_assigned(fxt_project_id, dataset_view_id=complete_view) is True
        assert fxt_subset_service.has_all_subsets_assigned(fxt_project_id, dataset_view_id=partial_view) is False


class TestSubsetServiceVideoGroups:
    """Video frames expose their parent video as group, and previously assigned
    videos yield subset pins for incremental annotation."""

    @staticmethod
    def _add_video_with_frames(
        db_session: Session,
        project_id: str,
        label: LabelDB,
        frame_subsets: list[DatasetItemSubset],
    ) -> str:
        from datetime import datetime

        from app.db.schema import DatasetItemLabelDB, MediaDB

        created = datetime.fromisoformat("2025-02-01T00:00:00Z")
        video = MediaDB(
            id=str(uuid4()),
            type="video",
            name="test_video",
            format="mp4",
            size=4096,
            width=1920,
            height=1080,
            fps=30.0,
            frame_count=len(frame_subsets),
            project_id=project_id,
            created_at=created,
        )
        db_session.add(video)
        db_session.flush()
        for frame_index, subset in enumerate(frame_subsets):
            frame = MediaDB(
                id=str(uuid4()),
                type="video_frame",
                name=f"frame_{frame_index}",
                format="mp4",
                size=1024,
                width=1920,
                height=1080,
                project_id=project_id,
                video_id=video.id,
                frame_index=frame_index,
                created_at=created,
            )
            db_session.add(frame)
            db_session.flush()
            db_session.add(DatasetItemDB(id=frame.id, subset=str(subset), project_id=project_id, created_at=created))
            db_session.add(DatasetItemLabelDB(dataset_item_id=frame.id, label_id=label.id))
        db_session.flush()
        return video.id

    def test_unassigned_video_frames_carry_group_id(
        self,
        fxt_project_id: UUID,
        fxt_subset_service: SubsetService,
        fxt_db_labels: list[LabelDB],
        db_session: Session,
    ) -> None:
        video_id = self._add_video_with_frames(
            db_session,
            str(fxt_project_id),
            fxt_db_labels[0],
            [DatasetItemSubset.UNASSIGNED] * 3,
        )

        items = fxt_subset_service.get_unassigned_items_with_labels(fxt_project_id)

        frame_groups = [item.group_id for item in items if item.group_id is not None]
        assert len(frame_groups) == 3
        assert all(group == UUID(video_id) for group in frame_groups)

    def test_pinned_group_subsets_prefers_majority_subset(
        self,
        fxt_project_id: UUID,
        fxt_subset_service: SubsetService,
        fxt_db_labels: list[LabelDB],
        db_session: Session,
    ) -> None:
        video_id = self._add_video_with_frames(
            db_session,
            str(fxt_project_id),
            fxt_db_labels[0],
            [
                DatasetItemSubset.TRAINING,
                DatasetItemSubset.TRAINING,
                DatasetItemSubset.VALIDATION,
                DatasetItemSubset.UNASSIGNED,
            ],
        )

        pins = fxt_subset_service.get_pinned_group_subsets(fxt_project_id)

        assert pins.get(UUID(video_id)) == DatasetItemSubset.TRAINING
