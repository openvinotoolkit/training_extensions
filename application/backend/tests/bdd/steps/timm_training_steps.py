# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import cast
from uuid import UUID

from behave import given, then, when
from behave.runner import Context
from requests import Session

from app.api.schemas import ModelView, ProjectView, StagedDatasetView
from app.api.schemas.jobs import JobView
from app.api.schemas.jobs.dataset_import import ImportDatasetMetadata
from app.api.schemas.jobs.training import TrainingMetadata
from app.models import TaskType, TrainingStatus
from app.supported_models.timm.manifest_provider import model_name_to_id
from tests.bdd.utils import (
    JobFailedError,
    download_file,
    import_dataset_as_new_project,
    prepare_dataset,
    train,
    upload_staged_dataset,
)

# Cache downloads across scenario runs to avoid re-fetching a multi-MB archive every time.
_DATASET_CACHE_DIR = Path(__file__).parent.parent / ".dataset_cache"


@given('a project "{project_name}" is created from the dataset archive at "{url}"')  # pyrefly: ignore
def step_project_from_dataset_url(context: Context, project_name: str, url: str) -> None:
    """Download a real dataset archive, stage it, and import it as a new classification project."""
    session = cast(Session, context.session)
    base_url = str(context.base_url)

    response = session.get(f"{base_url}/api/projects")
    assert response.status_code == 200, (
        f"Expected status code 200, got {response.status_code}, response: {response.text}"
    )
    existing_projects = [ProjectView.model_validate(proj) for proj in response.json()]
    existing_project = next((proj for proj in existing_projects if proj.name == project_name), None)
    if existing_project is not None:
        context.project = existing_project
        return

    archive_path = download_file(url, _DATASET_CACHE_DIR)
    staged_dataset_id = upload_staged_dataset(session, base_url, archive_path)

    prepare_dataset(session=session, base_url=base_url, staged_dataset_id=str(staged_dataset_id))

    response = session.get(f"{base_url}/api/staged_datasets/{staged_dataset_id}")
    assert response.status_code == 200, (
        f"Expected status code 200, got {response.status_code}, response: {response.text}"
    )
    staged_dataset = StagedDatasetView.model_validate(response.json())
    assert staged_dataset.ready_for_import, "Expected staged dataset to be ready for import"
    assert staged_dataset.metadata is not None, "Expected staged dataset metadata"
    labels = staged_dataset.metadata.labels

    job = import_dataset_as_new_project(
        session=session,
        base_url=base_url,
        project_name=project_name,
        staged_dataset_id=str(staged_dataset_id),
        labels=labels,
        task_type=TaskType.CLASSIFICATION,
        exclusive_labels=True,
    )
    project_id = cast(UUID, cast(ImportDatasetMetadata, job.metadata).project_id)

    response = session.get(f"{base_url}/api/projects/{project_id}")
    assert response.status_code == 200, (
        f"Expected status code 200, got {response.status_code}, response: {response.text}"
    )
    context.project = ProjectView.model_validate(response.json())


@given(
    'the training configuration for model architecture "{model_id}" is set to {max_epochs:d} epochs'  # pyrefly: ignore
)
def step_set_max_epochs(context: Context, model_id: str, max_epochs: int) -> None:
    """Cap the number of training epochs for the given model architecture, to keep the smoke test fast."""
    project = cast(ProjectView, context.project)
    session = cast(Session, context.session)
    response = session.patch(
        f"{context.base_url}/api/projects/{project.id}/training_configuration",
        params={"model_architecture_id": model_name_to_id(model_id)},
        json={"training.max_epochs": max_epochs},
    )
    assert response.status_code == 200, (
        f"Expected status code 200, got {response.status_code}, response: {response.text}"
    )


@when('I train timm model architecture "{model_id}" on device "{device}"')  # pyrefly: ignore
def step_train_timm_model(context: Context, model_id: str, device: str) -> None:
    """Train a timm model architecture on the given device."""
    project = cast(ProjectView, context.project)
    session = cast(Session, context.session)
    try:
        context.job = train(
            session=session,
            base_url=str(context.base_url),
            project_id=project.id,
            model_id=model_name_to_id(model_id),
            device=device,
        )
    except JobFailedError as exc:
        context.failures.append({"model_id": model_id, "stacktrace": exc.job.error or "<no stacktrace captured>"})
        raise


@then('the trained model has a "{variant_format}" variant with a positive weights size')  # pyrefly: ignore
def step_trained_model_has_variant(context: Context, variant_format: str) -> None:
    """Verify the resulting model revision is marked successful and has a usable artifact."""
    project = cast(ProjectView, context.project)
    session = cast(Session, context.session)
    job = cast(JobView, context.job)
    model_id = cast(TrainingMetadata, job.metadata).model.id

    response = session.get(f"{context.base_url}/api/projects/{project.id}/models/{model_id}")
    assert response.status_code == 200, (
        f"Expected status code 200, got {response.status_code}, response: {response.text}"
    )
    model = ModelView.model_validate(response.json())

    assert model.training_info.status == TrainingStatus.SUCCESSFUL, (
        f"Expected training status '{TrainingStatus.SUCCESSFUL}', got '{model.training_info.status}'"
    )
    variant = next((v for v in model.variants if v.format == variant_format), None)
    assert variant is not None, (
        f"Expected a '{variant_format}' variant, got formats: {[v.format for v in model.variants]}"
    )
    assert variant.weights_size > 0, f"Expected positive weights size for '{variant_format}' variant, got 0"
