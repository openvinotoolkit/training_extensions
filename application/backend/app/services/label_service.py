# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from loguru import logger
from sqlalchemy.orm import Session

from app.db.schema import LabelDB
from app.models import Label
from app.models.label import LabelReference, LabelUpdateInfo
from app.models.project import Project
from app.models.task import Task, TaskType
from app.repositories import DatasetItemRepository, LabelRepository
from app.repositories.base import PrimaryKeyIntegrityError, UniqueConstraintIntegrityError
from app.utils.color import random_color

from . import BaseSessionManagedService
from .base import ResourceNotFoundError, ResourceType, ResourceWithIdAlreadyExistsError
from .event.event_bus import EventBus, EventType


class DuplicateLabelsError(Exception):
    """Exception raised when label with duplicated names or hotkeys are being stored."""

    def __init__(self):
        super().__init__("Either label names or hotkeys have duplicates")


class LabelService(BaseSessionManagedService):
    def __init__(self, db_session: Session | None = None, event_bus: EventBus | None = None):
        super().__init__(db_session)
        self._event_bus: EventBus | None = event_bus

    def create_label(
        self, project_id: UUID, name: str, color: str | None, hotkey: str | None, label_id: UUID | None = None
    ) -> Label:
        label_repo = LabelRepository(str(project_id), self.db_session)
        try:
            db_label = label_repo.save(
                LabelDB(
                    id=str(label_id) if label_id is not None else None,
                    project_id=str(project_id),
                    name=name,
                    color=color if color else random_color(),
                    hotkey=hotkey,
                )
            )
            return Label.model_validate(db_label)
        except UniqueConstraintIntegrityError:
            raise DuplicateLabelsError
        except PrimaryKeyIntegrityError:
            raise ResourceWithIdAlreadyExistsError(ResourceType.LABEL, str(label_id))

    def list_all(self, project_id: UUID) -> list[Label]:
        label_repo = LabelRepository(str(project_id), self.db_session)
        db_labels = label_repo.list_all()
        return [Label.model_validate(db_label) for db_label in db_labels]

    def list_ids(self, project_id: UUID) -> list[UUID]:
        label_repo = LabelRepository(str(project_id), self.db_session)
        db_ids = label_repo.list_ids()
        return [UUID(db_id) for db_id in db_ids]

    def update_labels(
        self,
        project: Project,
        labels_to_add: list[Label],
        labels_to_remove: list[LabelReference],
        labels_to_edit: list[LabelUpdateInfo],
    ) -> list[Label]:
        """
        Update labels for a given project by adding, removing, and editing labels.

        Validates that the resulting number of labels satisfies project task constraints
        (e.g., multi-class classification requires at least two labels, and every project
        requires at least one label). Also validates that labels to remove or edit exist
        in the project before applying changes.

        If any label colour is effectively changed, an ``INFERENCE_PARAMS_CHANGED`` event is emitted once the
        transaction commits, so that a running inference pipeline refreshes the colours it uses to
        render predictions without having to reload the model.

        Args:
            project (Project): The project whose labels to update.
            labels_to_add (list[Label]): Labels to be added to the project.
            labels_to_remove (list[LabelReference]): Labels to be removed from the project.
            labels_to_edit (list[LabelUpdateInfo]): Labels within the project to be edited.

        Returns:
            list[Label]: The full list of labels for the project after all updates have been applied.

        Raises:
            ValueError: If the resulting number of labels violates project task constraints.
            ResourceNotFoundError: If any labels to remove or edit do not exist in the project.
            DuplicateLabelsError: If any label names or hotkeys conflict with existing labels.
            ResourceWithIdAlreadyExistsError: If a label to add has an ID that already exists.
        """

        # Validate minimal number of labels satisfies project task constraints
        existing_labels = {label.id: label for label in self.list_all(project_id=project.id)}
        existing_ids = list(existing_labels)
        new_number_of_labels = len(existing_ids) - len(labels_to_remove) + len(labels_to_add)
        if (
            project.task.task_type is TaskType.CLASSIFICATION
            and project.task.exclusive_labels
            and new_number_of_labels < 2
        ):
            raise ValueError(
                f"Multi-class classification requires at least two labels, but after this label update the total "
                f"number of labels is {new_number_of_labels}."
            )
        if new_number_of_labels < 1:
            raise ValueError(
                f"A project requires at least one label, but after this label update the total number of labels is "
                f"{new_number_of_labels}."
            )

        # Validate labels to remove or edit exist in project
        if missing_ids_to_remove := [label.id for label in labels_to_remove if label.id not in existing_ids]:
            raise ResourceNotFoundError(
                resource_type=ResourceType.LABEL,
                resource_id=str(missing_ids_to_remove[0]),
                message="One or more labels to remove do not exist in the project",
            )
        if missing_ids_to_edit := [label.id for label in labels_to_edit if label.id not in existing_ids]:
            raise ResourceNotFoundError(
                resource_type=ResourceType.LABEL,
                resource_id=str(missing_ids_to_edit[0]),
                message="One or more labels to edit do not exist in the project",
            )

        for label_to_edit in labels_to_edit:
            self._update_label(
                project_id=project.id,
                label_id=label_to_edit.id,
                new_name=label_to_edit.new_name,
                new_color=label_to_edit.new_color,
                new_hotkey=label_to_edit.new_hotkey,
            )
        for label_to_remove in labels_to_remove:
            self._delete_label(project_id=project.id, label_id=label_to_remove.id, task=project.task)
        for label_to_add in labels_to_add:
            self.create_label(
                project_id=project.id,
                label_id=label_to_add.id,
                name=label_to_add.name,
                color=label_to_add.color,
                hotkey=label_to_add.hotkey,
            )

        if any(
            label_to_edit.new_color is not None and existing_labels[label_to_edit.id].color != label_to_edit.new_color
            for label_to_edit in labels_to_edit
        ):
            self._notify_label_colors_changed(project_id=project.id)

        return self.list_all(project_id=project.id)

    def _notify_label_colors_changed(self, project_id: UUID) -> None:
        """Signal, after the transaction commits, that the label colours of a project changed.

        The inference worker caches the label colours used to render predictions; without this
        notification the overlay would keep the previous colours until the model or the pipeline is
        reloaded for some other reason.
        """
        if self._event_bus is None:
            return
        logger.debug("Label colors changed for project '{}'; notifying inference pipeline", project_id)
        self._event_bus.emit_event_after_commit(self.db_session, EventType.INFERENCE_PARAMS_CHANGED)

    def _update_label(
        self, project_id: UUID, label_id: UUID, new_name: str | None, new_color: str | None, new_hotkey: str | None
    ) -> Label:
        label_repo = LabelRepository(str(project_id), self.db_session)
        try:
            db_label = label_repo.get_by_id(str(label_id))
            if db_label is None:
                raise ResourceNotFoundError(ResourceType.LABEL, str(label_id))
            db_label = label_repo.update(
                LabelDB(
                    id=db_label.id,
                    project_id=str(project_id),
                    name=new_name,
                    color=new_color,
                    hotkey=new_hotkey,
                )
            )
            return Label.model_validate(db_label)
        except UniqueConstraintIntegrityError:
            raise DuplicateLabelsError

    def _delete_label(self, project_id: UUID, label_id: UUID, task: Task) -> None:
        """Delete a label and clean up all annotations that reference it.

        For each dataset item that uses this label:
        - Remove the label from all annotation shapes.
        - Remove shapes that have no labels left.
        - If no shapes remain, set annotation_data=[].
        - For multi-class classification (which doesn't support empty labels),
          set annotation_data=None and user_reviewed=False.
        """
        dataset_item_repo = DatasetItemRepository(project_id=str(project_id), db=self.db_session)
        label_id_str = str(label_id)

        # Find all dataset items that reference this label
        affected_items = dataset_item_repo.find_items_by_label_id(label_id_str)

        for item in affected_items:
            if item.annotation_data is None:
                continue

            # Build a fresh list of annotations with the deleted label removed.
            # Shapes that have no remaining labels are dropped entirely.
            updated_annotations = []
            for annotation in item.annotation_data:
                filtered_labels = [lbl for lbl in annotation.get("labels", []) if lbl.get("id") != label_id_str]
                if filtered_labels:
                    updated_annotations.append({**annotation, "labels": filtered_labels})

            if updated_annotations:
                dataset_item_repo.set_annotation_data(
                    obj_id=item.id,
                    annotation_data=updated_annotations,
                    user_reviewed=item.user_reviewed,
                    prediction_model_id=item.prediction_model_id,
                )
            elif task.is_multiclass:
                dataset_item_repo.delete_annotation_data(obj_id=item.id)
            else:
                dataset_item_repo.set_annotation_data(
                    obj_id=item.id,
                    annotation_data=[],
                    user_reviewed=item.user_reviewed,
                    prediction_model_id=item.prediction_model_id,
                )

        # Remove label references from the dataset_items_labels join table
        dataset_item_repo.delete_label_from_items(label_id_str)

        # Delete the label itself
        label_repo = LabelRepository(str(project_id), self.db_session)
        if not label_repo.delete(label_id_str):
            raise ResourceNotFoundError(ResourceType.LABEL, label_id_str)
