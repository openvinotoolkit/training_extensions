# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Run through the CLI: uv run app/cli.py export-project --project-id <id> --output <archive_path>

import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger
from sqlalchemy import Engine, MetaData, Table, inspect, select

from app.db.engine import db_engine
from app.db.import_export.common import get_database_schema_version, get_project_files_path, validate_project_id


def _query_dataset_items_labels(engine: Engine, metadata: MetaData, project_id: str):
    """Join through dataset_items to get project-related labels."""
    dil = Table("dataset_items_labels", metadata, autoload_with=engine)
    di = Table("dataset_items", metadata, autoload_with=engine)

    return select(dil).join(di, dil.c.dataset_item_id == di.c.id).where(di.c.project_id == project_id)


def _query_dataset_view_items(engine: Engine, metadata: MetaData, project_id: str):
    """Join through dataset_views to get project-related dataset view item assignments."""
    dvi = Table("dataset_view_items", metadata, autoload_with=engine)
    dv = Table("dataset_views", metadata, autoload_with=engine)

    return select(dvi).join(dv, dvi.c.dataset_view_id == dv.c.id).where(dv.c.project_id == project_id)


def _query_metric_scores(engine: Engine, metadata: MetaData, project_id: str):
    """Join through evaluations, model_variants, and model_revisions to get project-related metric scores."""
    ms = Table("metric_scores", metadata, autoload_with=engine)
    e = Table("evaluations", metadata, autoload_with=engine)
    mv = Table("model_variants", metadata, autoload_with=engine)
    mr = Table("model_revisions", metadata, autoload_with=engine)

    return (
        select(ms)
        .join(e, ms.c.evaluation_id == e.c.id)
        .join(mv, e.c.model_variant_id == mv.c.id)
        .join(mr, mv.c.model_revision_id == mr.c.id)
        .where(mr.c.project_id == project_id)
    )


def _query_evaluations(engine: Engine, metadata: MetaData, project_id: str):
    """Join through model_variants and model_revisions to get project-related evaluations."""
    e = Table("evaluations", metadata, autoload_with=engine)
    mv = Table("model_variants", metadata, autoload_with=engine)
    mr = Table("model_revisions", metadata, autoload_with=engine)

    return (
        select(e)
        .join(mv, e.c.model_variant_id == mv.c.id)
        .join(mr, mv.c.model_revision_id == mr.c.id)
        .where(mr.c.project_id == project_id)
    )


def _query_model_variants(engine: Engine, metadata: MetaData, project_id: str):
    """Join through model_revisions to get project-related model variants."""
    mv = Table("model_variants", metadata, autoload_with=engine)
    mr = Table("model_revisions", metadata, autoload_with=engine)

    return select(mv).join(mr, mv.c.model_revision_id == mr.c.id).where(mr.c.project_id == project_id)


def _query_video_frames(engine: Engine, metadata: MetaData, project_id: str):
    """Join through media (video_id) to get project-related video frames."""
    vf = Table("video_frames", metadata, autoload_with=engine)
    media = Table("media", metadata, autoload_with=engine)

    return select(vf).join(media, vf.c.video_id == media.c.id).where(media.c.project_id == project_id)


# Mapping of table names to their query strategy to fetch the relevant records for the given project.
# Strategies:
# - None: the table is excluded from export
# - "project_id": the table has a 'project_id' column, so the query can directly filter on that
# - callable: custom query function, which may involve joins to other tables
QUERY_STRATEGY_BY_TABLE = {
    # Excluded tables (not owned by project)
    "alembic_version": None,
    "sources": None,
    "sinks": None,
    # Tables with direct project_id column
    "projects": "project_id",
    "pipelines": "project_id",
    "model_revisions": "project_id",
    "dataset_revisions": "project_id",
    "dataset_items": "project_id",
    "media": "project_id",
    "labels": "project_id",
    "training_configurations": "project_id",
    "dataset_views": "project_id",
    # Tables requiring joins
    "dataset_items_labels": _query_dataset_items_labels,
    "dataset_view_items": _query_dataset_view_items,
    "model_variants": _query_model_variants,
    "evaluations": _query_evaluations,
    "metric_scores": _query_metric_scores,
    "video_frames": _query_video_frames,
}


def _serialize_value(value: Any) -> Any:
    """
    Serialize a value for JSON export. Values that are not JSON-serializable (like datetime) are converted to strings.
    """
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _get_project_filter_query(engine: Engine, metadata: MetaData, table_name: str, project_id: str):
    """Generate query for tables that have a 'project_id' column."""
    table = Table(table_name, metadata, autoload_with=engine)

    if table_name == "projects":
        return select(table).where(table.c.id == project_id)
    return select(table).where(table.c.project_id == project_id)


def _export_table(engine: Engine, metadata: MetaData, table_name: str, project_id: str) -> list[dict] | None:
    """Export all rows from a table that belong to the given project."""
    query_strategy = QUERY_STRATEGY_BY_TABLE.get(table_name)

    if query_strategy is None:
        return None  # Excluded table

    if query_strategy == "project_id":
        query = _get_project_filter_query(engine, metadata, table_name, project_id)
    elif callable(query_strategy):
        query = query_strategy(engine, metadata, project_id)
    else:
        raise ValueError(f"Invalid configuration for table {table_name}")

    with engine.connect() as conn:
        result = conn.execute(query)
        columns = result.keys()
        return [dict(zip(columns, row)) for row in result.fetchall()]


def _validate_output_path(output_path: Path) -> None:
    """Validate that the output path is valid for writing the zip archive."""
    if output_path.suffix.lower() != ".zip":
        raise ValueError(f"Output file must have .zip extension: {output_path}")
    if not output_path.parent.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_path.parent}")
    if output_path.exists():
        raise FileExistsError(f"Output archive already exists: {output_path}")


def _export_project_files(project_files_path: Path, dest_path: Path) -> dict:
    """
    Copy project files to the destination path.
    Returns metadata about the copied files.
    """
    if not project_files_path.exists():
        raise FileNotFoundError(f"Project files not found at the expected path: {project_files_path}")

    if not project_files_path.is_dir():
        raise ValueError(f"Project files path is not a directory: {project_files_path}")

    try:
        shutil.copytree(project_files_path, dest_path)
    except Exception as e:
        raise RuntimeError(f"Failed to copy project files: {str(e)}") from e

    # Count files and calculate total size
    file_count = sum(1 for _ in dest_path.rglob("*") if _.is_file())
    total_size = sum(f.stat().st_size for f in dest_path.rglob("*") if f.is_file())

    return {
        "file_count": file_count,
        "total_size_bytes": total_size,
    }


def export_project(project_id: str, output_archive: str) -> None:  # noqa: C901, PLR0915
    """Export all project data to a zip archive."""
    # Validate project_id is a valid UUID
    validate_project_id(project_id)

    # Get project files path from settings
    project_files_path = get_project_files_path(project_id)

    # Validate and prepare output path
    output_path = Path(output_archive)
    _validate_output_path(output_path)

    # Use the shared database engine and determine the schema version
    engine = db_engine
    metadata = MetaData()
    db_schema_version = get_database_schema_version()
    logger.info("Database version: {}", db_schema_version)

    # Create temporary directory for export
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        tables_path = temp_path / "tables"
        tables_path.mkdir(parents=True, exist_ok=True)
        binaries_path = temp_path / "binaries"

        # Get all tables from the database
        try:
            inspector = inspect(engine)
            db_tables = set(inspector.get_table_names())  # pyrefly: ignore[missing-attribute]
        except Exception as e:
            raise RuntimeError(f"Failed to inspect database: {str(e)}") from e

        # Check for unmapped tables
        unmapped_tables = db_tables - set(QUERY_STRATEGY_BY_TABLE.keys())
        if unmapped_tables:
            raise ValueError(f"Found unmapped tables in database: {unmapped_tables}. Please add them to TABLE_CONFIG.")

        # Verify project exists
        projects_table = Table("projects", metadata, autoload_with=engine)
        with engine.connect() as conn:
            try:
                result = conn.execute(select(projects_table.c.id).where(projects_table.c.id == project_id))
                if not result.fetchone():
                    raise ValueError(f"Project with id '{project_id}' not found in database")
            except Exception as e:
                raise ValueError(f"Failed to verify project existence: {str(e)}") from e

        exported_tables = {}

        for table_name in QUERY_STRATEGY_BY_TABLE:
            if table_name not in db_tables:
                logger.warning("Table '{}' in config but not in database, skipping", table_name)
                continue

            try:
                rows = _export_table(engine, metadata, table_name, project_id)
            except Exception as e:
                logger.error("Failed to export table '{}': {}", table_name, str(e))
                raise

            if rows is None:
                logger.info("Skipping excluded table: {}", table_name)
                continue

            # Serialize datetime values
            serialized_rows = [{k: _serialize_value(v) for k, v in row.items()} for row in rows]

            # Save to tables/ subfolder
            output_file = tables_path / f"{table_name}.json"
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(serialized_rows, f, indent=2)
            except Exception as e:
                raise RuntimeError(f"Failed to write table file {table_name}.json: {str(e)}") from e

            exported_tables[table_name] = len(serialized_rows)
            logger.info("Exported {} rows from '{}'", len(serialized_rows), table_name)

        # Export project files (binaries)
        logger.info("Exporting project files from: {}", project_files_path)
        binaries_info = _export_project_files(project_files_path, binaries_path)
        logger.info(
            "Exported {} files ({} bytes) from project folder",
            binaries_info["file_count"],
            binaries_info["total_size_bytes"],
        )

        # Write manifest at root level
        logger.info("Writing manifest.json with export metadata")
        manifest = {
            "project_id": project_id,
            "export_date": datetime.now(UTC).isoformat(),
            "database_version": db_schema_version,
            "tables": exported_tables,
            "binaries": binaries_info,
        }
        try:
            with open(temp_path / "manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            raise RuntimeError(f"Failed to write manifest.json: {str(e)}") from e

        # Create zip archive
        logger.info("Creating zip archive at: {}; this operation may take a while for large projects", output_path)
        try:
            archive_base = str(output_path.with_suffix(""))
            shutil.make_archive(archive_base, "zip", temp_path)
        except Exception as e:
            raise ValueError(f"Failed to create zip archive: {str(e)}") from e

        logger.success("Export complete. Archive saved to: {}", output_path)
