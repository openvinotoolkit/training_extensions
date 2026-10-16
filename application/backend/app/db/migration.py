# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Database migration management"""

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime import migration
from alembic.script import ScriptDirectory
from alembic.script.revision import ResolutionError
from alembic.util.exc import CommandError
from loguru import logger
from sqlalchemy import text

from app.db import db_engine
from app.settings import Settings


class RevisionNotFoundError(Exception):
    """Raised when the current revision is not found in Alembic history."""


class MigrationFatalError(Exception):
    """Raised when a database migration fails in a way a restart cannot fix.

    This signals a fatal, non-restartable condition (e.g. a failed Alembic
    upgrade or an unrecognized schema revision). Callers at the process boundary
    should surface it and exit with a dedicated exit code instead of retrying.

    Attributes:
        backup_path: Path to the pre-migration database backup that must be
            restored to recover the previous version's data, or ``None`` when no
            backup was taken (e.g. a fresh install with no prior database).
    """

    def __init__(self, message: str, backup_path: Path | None = None) -> None:
        super().__init__(message)
        self.backup_path = backup_path


class MigrationManager:
    """Manages database connections and migrations"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.__ensure_data_directory()

    def __ensure_data_directory(self) -> None:
        """Ensure the data directory exists"""
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def check_connection() -> bool:
        """Check if database connection is working"""
        try:
            with db_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error("Database connection check failed: {}", e)
            return False

    def get_alembic_config(self) -> Config:
        """Get Alembic configuration"""
        alembic_cfg = Config(self.settings.alembic_config_path)
        alembic_cfg.set_main_option("script_location", self.settings.alembic_script_location)
        alembic_cfg.set_main_option("sqlalchemy.url", self.settings.database_url)
        return alembic_cfg

    @property
    def database_path(self) -> Path:
        """Filesystem path of the SQLite database file."""
        return self.settings.data_dir / self.settings.database_file

    def _backup_database(self) -> Path | None:
        """Create a consistent physical backup of the SQLite database file.

        Taken right before a migration so the previous version's data can be
        recovered if the upgrade fails or produces a broken schema. This matters
        especially for MSIX upgrades, where the old package (and its Alembic
        ``downgrade`` scripts) is gone after the update, making a file-level
        backup the only reliable rollback path.

        SQLite's online backup API is used (rather than a plain file copy) so any
        data still in the WAL/SHM sidecar files is folded into a single, consistent
        backup file.

        Returns:
            The path to the created backup, or ``None`` if there is no database
            file to back up yet (fresh install).
        """
        db_path = self.database_path
        if not db_path.exists():
            logger.info("No existing database file to back up (fresh install)")
            return None

        timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
        backup_path = db_path.with_name(f"{db_path.name}.{timestamp}.bak")

        logger.info("Backing up database to {} before migration...", backup_path)
        with closing(sqlite3.connect(str(db_path))) as source, closing(sqlite3.connect(str(backup_path))) as dest:
            source.backup(dest)

        logger.info("✓ Database backup created at {}", backup_path)
        return backup_path

    @staticmethod
    def _remove_backup(backup_path: Path) -> None:
        """Delete a database backup file once it is no longer needed.

        Called after a successful migration so backups don't accumulate on disk.
        Failure to remove the backup is logged but not fatal.
        """
        try:
            backup_path.unlink(missing_ok=True)
            logger.info("✓ Removed database backup {} after successful migration", backup_path)
        except OSError as e:
            logger.warning("Could not remove database backup {}: {}", backup_path, e)

    def run_migrations(self, backup_path: Path | None = None) -> bool:
        """Run database migrations.

        Args:
            backup_path: Path to the pre-migration database backup, attached to
                any raised :class:`MigrationFatalError` so the recovery point is
                reported at the process boundary.

        Returns:
            ``True`` when the migration completes successfully.

        Raises:
            MigrationFatalError: if the migration fails. This is a fatal,
                non-restartable condition.
        """
        try:
            logger.info("Running database migrations...")
            alembic_cfg = self.get_alembic_config()
            command.upgrade(alembic_cfg, "head")
            logger.info("✓ Database migrations completed successfully")
            return True
        except Exception as e:
            logger.error("✗ Database migration failed: {}", e)
            raise MigrationFatalError(f"Database migration failed: {e}", backup_path=backup_path) from e

    def check_migration_status(self) -> tuple[bool, str]:
        """Check if database needs migration"""
        try:
            alembic_cfg = self.get_alembic_config()
            script = ScriptDirectory.from_config(alembic_cfg)
            current_head = script.get_current_head()

            with db_engine.connect() as conn:
                context = migration.MigrationContext.configure(conn)
                current_rev = context.get_current_revision()

            # Check if current_rev is in Alembic's tracked revisions
            if current_rev:
                try:
                    script.get_revision(current_rev)
                except (CommandError, ResolutionError) as e:
                    raise RevisionNotFoundError(
                        f"Current revision '{current_rev}' not found in Alembic history. Please, recreate the database."
                    ) from e

            needs_migration = current_rev != current_head
            status = f"Current: {current_rev or 'None'}, Head: {current_head or 'None'}"

            return needs_migration, status

        except RevisionNotFoundError:
            raise
        except Exception as e:
            logger.warning("Could not check migration status: {}", e)
            return True, "Unknown - assuming migration needed"

    def initialize_database(self) -> bool:
        """Initialize database with migrations if needed.

        Returns:
            ``True`` if the database is ready to use. ``False`` indicates a
            transient failure (e.g. the database is temporarily unreachable) that
            may succeed on a subsequent restart.

        Raises:
            MigrationFatalError: for fatal, non-restartable failures such as a
                failed migration or an unrecognized schema revision.
        """
        try:
            # Ensure data directory exists
            self.__ensure_data_directory()

            # Check if we can connect
            if not self.check_connection():
                logger.error("Cannot connect to database")
                return False

            # Check migration status
            needs_migration, status = self.check_migration_status()
            logger.info("Migration status: {}", status)

            if needs_migration:
                logger.info("Database needs migration")
                backup_path = self._backup_database()
                # run_migrations() raises MigrationFatalError on failure; the
                # backup is intentionally retained for manual recovery and its
                # location is attached to the raised error.
                self.run_migrations(backup_path=backup_path)
                if backup_path is not None:
                    self._remove_backup(backup_path)
                return True
            logger.info("Database is up to date")
            return True

        except MigrationFatalError:
            # Already fatal and non-restartable; propagate to the process boundary.
            raise
        except RevisionNotFoundError as e:
            logger.error("Revision not found: {}", e)
            raise MigrationFatalError(f"Unrecognized database schema revision: {e}") from e
        except Exception as e:
            logger.error("Database initialization failed: {}", e)
            return False
