# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path
from uuid import UUID

from sqlalchemy import text

from app.db.engine import db_engine
from app.settings import get_settings


class InvalidArchiveError(ValueError):
    """Exception raised when the input archive is invalid or does not conform to expected structure."""


class DatabaseVersionMismatchError(ValueError):
    """Exception raised when database version in archive doesn't match target database."""


class ProjectAlreadyExistsError(ValueError):
    """Exception raised when trying to import a project that already exists."""


def validate_project_id(project_id: str) -> None:
    """Validate that project_id is a valid UUID."""
    if not project_id or not project_id.strip():
        raise ValueError("Project ID cannot be empty")

    try:
        UUID(project_id)
    except ValueError:
        raise ValueError(f"Invalid project ID format: '{project_id}' is not a valid UUID")


def get_data_folder_path() -> Path:
    """Get the data folder path from settings."""
    settings = get_settings()
    return settings.data_dir


def get_project_files_path(project_id: str) -> Path:
    """Get the path to the project files folder."""
    data_folder = get_data_folder_path()
    return data_folder / "projects" / project_id


def get_database_schema_version() -> str:
    """Get the current database schema version from alembic_version table."""
    with db_engine.connect() as conn:
        result = conn.execute(text("SELECT version_num FROM alembic_version"))
        row = result.fetchone()
        if row:
            return row[0]
        raise ValueError("No version found in 'alembic_version' table")


def copy_folder_recursively(source_path: Path, dest_path: Path) -> tuple[int, int]:
    """
    Copy all files from a source folder to a destination folder, preserving directory structure.
    Returns a tuple of (number of files copied, total size in bytes).
    """
    if not source_path.exists():
        raise FileNotFoundError(f"Source path not found: {source_path}")

    if not source_path.is_dir():
        raise ValueError(f"Source path is not a directory: {source_path}")

    try:
        shutil.copytree(source_path, dest_path)
    except Exception as e:
        raise RuntimeError(f"Failed to copy files: {str(e)}") from e

    # Count files and calculate total size
    file_count = sum(1 for _ in dest_path.rglob("*") if _.is_file())
    total_size = sum(f.stat().st_size for f in dest_path.rglob("*") if f.is_file())

    return file_count, total_size
