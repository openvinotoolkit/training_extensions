#  Copyright (C) 2026 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

# Run through the CLI: uv run app/cli.py import-project --input <archive_path>

import json
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

from loguru import logger
from sqlalchemy import Connection, MetaData, Table, inspect

from app.db.engine import db_engine
from app.db.import_export.common import (
    DatabaseVersionMismatchError,
    InvalidArchiveError,
    ProjectAlreadyExistsError,
    copy_folder_recursively,
    get_database_schema_version,
    get_project_files_path,
    validate_project_id,
)

# Import order matters due to foreign key constraints
# Tables are imported in this order to satisfy dependencies
IMPORT_ORDER = [
    "projects",
    "labels",
    "training_configurations",
    "media",
    "dataset_views",
    "dataset_revisions",
    "model_revisions",
    "model_variants",
    "dataset_items",
    "dataset_items_labels",
    "dataset_view_items",
    "evaluations",
    "metric_scores",
    "pipelines",
]


def _parse_datetime(value: str | datetime | None) -> datetime | None:
    """Parse ISO format datetime string back to datetime object."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def _deserialize_row(row: dict, columns_info: dict) -> dict:
    """Convert string values back to appropriate types."""
    result = {}
    for key, value in row.items():
        col_type = columns_info.get(key, {}).get("type", "")
        col_type_str = str(col_type).upper()
        if "DATETIME" in col_type_str:
            result[key] = _parse_datetime(value)
        else:
            result[key] = value
    return result


def _get_table_columns_info(table_name: str) -> dict:
    """Get column information for a table."""
    inspector = inspect(db_engine)
    columns = inspector.get_columns(table_name)  # pyrefly: ignore[missing-attribute]
    return {col["name"]: col for col in columns}


def _import_table(conn: Connection, metadata: MetaData, table_name: str, rows: list[dict], columns_info: dict) -> int:
    """Import rows into a table using SQLAlchemy 2.0 syntax within an existing connection/transaction."""
    if not rows:
        return 0

    table = Table(table_name, metadata, autoload_with=conn)

    for row in rows:
        deserialized = _deserialize_row(row, columns_info)
        # Use SQLAlchemy insert with on_conflict_do_update for "INSERT OR REPLACE" behavior
        from sqlalchemy.dialects.sqlite import insert

        stmt = insert(table).values(**deserialized)
        # Get primary key columns for the conflict target
        pk_cols = [col.name for col in table.primary_key.columns]
        update_dict = {k: v for k, v in deserialized.items() if k not in pk_cols}
        if update_dict:
            stmt = stmt.on_conflict_do_update(index_elements=pk_cols, set_=update_dict)
        else:
            stmt = stmt.on_conflict_do_nothing()
        conn.execute(stmt)

    return len(rows)


def _validate_archive_file(input_archive: str) -> None:
    """Validate that archive file exists and is a zip file."""
    archive_path = Path(input_archive)
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive file not found: {input_archive}")
    if not archive_path.is_file():
        raise ValueError(f"Archive path is not a file: {input_archive}")
    if archive_path.suffix.lower() != ".zip":
        raise ValueError(f"Invalid archive format (expected .zip): {input_archive}")


def _cleanup_project_files(project_files_path: Path) -> None:
    """Remove project files folder if it exists (used for rollback on error)."""
    if project_files_path.exists():
        try:
            shutil.rmtree(project_files_path)
            logger.info("Cleaned up project files at: {}", project_files_path)
        except Exception as e:
            logger.error("Failed to clean up project files at {}: {}", project_files_path, str(e))


def import_project(input_archive: str, allow_mismatching_db_schema: bool = False) -> None:  # noqa: C901, PLR0912, PLR0915
    """Import project data from a zip archive into the database.

    Args:
        input_archive: Path to the zip archive containing project data.
        allow_mismatching_db_schema: If True, bypasses database schema version check.
            Use with caution as this may cause import errors or data corruption.
    """
    # Validate archive file
    _validate_archive_file(input_archive)

    # Use the shared database engine and get schema version
    engine = db_engine
    target_db_schema_version = get_database_schema_version()
    logger.info("Target database schema version: {}", target_db_schema_version)

    # Extract zip to temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Extract archive
        try:
            with zipfile.ZipFile(input_archive, "r") as zip_ref:
                zip_ref.extractall(temp_path)
        except zipfile.BadZipFile as e:
            raise InvalidArchiveError(f"Invalid or corrupted zip archive: {str(e)}") from e
        except PermissionError as e:
            raise PermissionError(f"Permission denied reading archive: {input_archive}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to extract archive: {str(e)}") from e

        # Load and verify manifest
        manifest_file = temp_path / "manifest.json"
        if not manifest_file.exists():
            raise InvalidArchiveError("Manifest file not found in archive. Expected: manifest.json at root level")

        try:
            with open(manifest_file, encoding="utf-8") as f:
                manifest = json.load(f)
        except json.JSONDecodeError as e:
            raise InvalidArchiveError(f"Invalid manifest.json format: {str(e)}") from e
        except Exception as e:
            raise InvalidArchiveError(f"Failed to read manifest.json: {str(e)}") from e

        if "project_id" not in manifest:
            raise InvalidArchiveError("Manifest missing required 'project_id' field")

        project_id = manifest["project_id"]
        validate_project_id(project_id)

        # Check if project files folder already exists
        project_files_path = get_project_files_path(project_id)
        if project_files_path.exists():
            raise ProjectAlreadyExistsError(
                f"Project files folder already exists at: {project_files_path}. "
                f"Please remove or rename the existing folder before importing."
            )

        # Verify database version compatibility
        if "database_version" not in manifest:
            raise InvalidArchiveError(
                "Manifest missing required 'database_version' field for version compatibility check"
            )
        archive_db_version = manifest["database_version"]
        if archive_db_version != target_db_schema_version:
            if allow_mismatching_db_schema:
                logger.warning(
                    "Database version mismatch: archive has version '{}', but target database has version '{}'. "
                    "Proceeding anyway due to --force flag. This may cause errors or data corruption.",
                    archive_db_version,
                    target_db_schema_version,
                )
            else:
                export_date = manifest.get("export_date", "unknown date")
                raise DatabaseVersionMismatchError(
                    f"The archive data follows the schema with version '{archive_db_version}' (archive exported on "
                    f"'{export_date}'), however the target database has schema version '{target_db_schema_version}'. "
                    f"Usually, this means that the archive was exported from a different version of the application "
                    f"than the one you are currently running, and it may not be fully compatible. "
                    f"If you want to attempt the import anyway, run again with the '--force-import' flag."
                )
        else:
            logger.info("Database version verified: {}", archive_db_version)

        logger.info("Importing project: {}", project_id)

        # Verify all expected tables exist in the database
        try:
            inspector = inspect(engine)
            db_tables = set(inspector.get_table_names())  # pyrefly: ignore[missing-attribute]
        except Exception as e:
            raise ValueError(f"Failed to inspect database: {str(e)}") from e

        # Check for tables in manifest that don't exist in DB
        for table_name in manifest.get("tables", {}):
            if table_name not in db_tables:
                raise ValueError(f"Table '{table_name}' in manifest but not in database")

        # Verify tables path exists
        tables_path = temp_path / "tables"
        if not tables_path.exists():
            raise InvalidArchiveError("Tables directory not found in archive. Expected: tables/ at root level")

        # Preload all table data and column info before starting the transaction
        tables_data = {}
        tables_columns_info = {}
        for table_name in IMPORT_ORDER:
            json_file = tables_path / f"{table_name}.json"
            if not json_file.exists():
                logger.info("No data file for table '{}', skipping", table_name)
                continue

            try:
                with open(json_file, encoding="utf-8") as f:
                    tables_data[table_name] = json.load(f)
            except json.JSONDecodeError as e:
                raise InvalidArchiveError(f"Invalid JSON in {table_name}.json: {str(e)}") from e
            except Exception as e:
                raise InvalidArchiveError(f"Failed to read {table_name}.json: {str(e)}") from e

            tables_columns_info[table_name] = _get_table_columns_info(table_name)

        # Track whether files have been successfully copied (for rollback)
        files_copied = False

        try:
            # Import project files first (before DB transaction)
            # This way if file copy fails, we haven't touched the DB yet
            binaries_path = temp_path / "binaries"
            logger.info("Importing project files to: {}", project_files_path)
            project_files_path.parent.mkdir(parents=True, exist_ok=True)
            num_files_copied, num_bytes_copied = copy_folder_recursively(
                source_path=binaries_path, dest_path=project_files_path
            )
            files_copied = True
            logger.info("Imported {} files ({} bytes) to project folder", num_files_copied, num_bytes_copied)

            # Now perform all DB operations in a single atomic transaction
            metadata = MetaData()

            with engine.begin() as conn:
                for table_name in IMPORT_ORDER:
                    if table_name not in tables_data:
                        continue

                    rows = tables_data[table_name]
                    columns_info = tables_columns_info[table_name]

                    try:
                        count = _import_table(conn, metadata, table_name, rows, columns_info)
                        logger.info("Imported {} rows into '{}'", count, table_name)
                    except Exception as e:
                        raise RuntimeError(f"Failed to import table '{table_name}': {str(e)}") from e

                # Transaction commits automatically at end of 'with' block if no exception

        except Exception:
            # Rollback: clean up project files if they were copied
            if files_copied:
                logger.warning("Error occurred, rolling back project files import...")
                _cleanup_project_files(project_files_path)
            raise

        logger.success("Import complete. Project '{}' has been imported.", project_id)
