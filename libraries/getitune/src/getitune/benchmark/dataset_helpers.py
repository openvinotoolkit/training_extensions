# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for benchmark dataset preparation scripts.

Every preparation script in ``scripts/benchmark_datasets/`` follows the same
pattern: parse CLI args, download an archive, extract it, optionally transform
the data, and clean up.  This module provides reusable building blocks so that
individual scripts only need to define the dataset-specific logic.

Typical usage in a preparation script::

    #!/usr/bin/env python3
    from getitune.benchmark.dataset_helpers import (
        DatasetArgs,
        download,
        extract_archive,
        parse_args,
    )

    def main() -> None:
        args = parse_args(description="Prepare the pothole_tiny dataset.")

        archive = download(
            url="https://storage.geti.intel.com/test-data/pothole_tiny.tar.gz",
            dest_dir=args.archive_dir,
            filename="pothole_tiny.tar.gz",
        )

        extract_archive(archive, args.dest)

        # (optional) dataset-specific adjustments here …

        archive.unlink(missing_ok=True)
        print(f"Dataset '{args.name}' ready at {args.dest}")

    if __name__ == "__main__":
        main()

Datasets gated behind credentials (e.g. Kaggle) or too large/slow to download
on every run can use ``--raw-dir`` to skip the network fetch while still
running through the same transform/export pipeline — see
:func:`resolve_raw_source`. Kaggle-hosted datasets specifically should use
:func:`download_kaggle_dataset`, which gives a clear, actionable error when
credentials or ``kagglehub`` are missing instead of failing deep inside the
dependency.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parsed arguments
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetArgs:
    """Parsed CLI arguments common to all preparation scripts."""

    output_dir: Path
    """Root directory for dataset storage (``--output-dir``)."""

    name: str
    """Dataset name — determines the sub-directory (``--name``)."""

    raw_dir: Path | None = None
    """Optional pre-fetched raw-data location (``--raw-dir``).

    When set, preparation scripts should skip their own network download and
    read from this path instead (see :func:`resolve_raw_source`). This is how
    the catalog's ``raw_dir`` field (``catalog.py``) reaches a script: a
    dataset that requires credentials (e.g. Kaggle) or is otherwise slow to
    re-download can be fetched once, placed on disk, and reused indefinitely.
    """

    @property
    def dest(self) -> Path:
        """Final dataset directory: ``<output_dir>/<name>``."""
        return self.output_dir / self.name

    @property
    def archive_dir(self) -> Path:
        """Temporary directory for downloaded archives: ``<output_dir>/.archives``."""
        return self.output_dir / ".archives"


def parse_args(*, description: str = "Prepare a benchmark dataset.") -> DatasetArgs:
    """Parse the standard ``--output-dir`` / ``--name`` / ``--raw-dir`` CLI arguments.

    This should be the first call in every preparation script's ``main()``.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory for dataset storage.",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Dataset name (determines sub-directory).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help=(
            "Optional pre-fetched raw-data directory or archive file. When given, the "
            "script should skip its own network download and use this data instead "
            "(see resolve_raw_source())."
        ),
    )
    ns = parser.parse_args()
    return DatasetArgs(output_dir=ns.output_dir, name=ns.name, raw_dir=ns.raw_dir)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

_CHUNK_SIZE = 1 << 20  # 1 MiB


def download(url: str, dest_dir: Path, filename: str | None = None) -> Path:
    """Download *url* into *dest_dir* and return the local file path.

    Parameters
    ----------
    url:
        Remote URL to fetch.
    dest_dir:
        Directory to save the file in (created if missing).
    filename:
        Local file name.  Defaults to the last path segment of *url*.
    """
    if filename is None:
        filename = url.rsplit("/", 1)[-1]

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename

    print(f"Downloading {url} → {dest}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310  # nosec B310 - URLs come from trusted in-repo benchmark catalog
    return dest


# ---------------------------------------------------------------------------
# Archive extraction
# ---------------------------------------------------------------------------


def extract_archive(archive: Path, dest: Path, *, clean_dest: bool = True) -> Path:
    """Extract a ``.tar.gz``, ``.tar``, or ``.zip`` archive into *dest*.

    Parameters
    ----------
    archive:
        Path to the archive file.
    dest:
        Directory to extract into (created if missing).
    clean_dest:
        If ``True`` (default) and *dest* already exists, it is removed
        before extraction so the result is always a clean copy.

    Returns:
    -------
    Path
        The *dest* directory.
    """
    if clean_dest and dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            tf.extractall(dest, filter="data")
    elif zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)  # noqa: S202
    else:
        msg = f"Unsupported archive format: {archive}"
        raise ValueError(msg)

    return dest


# ---------------------------------------------------------------------------
# Raw-source resolution (credentialed / large datasets)
# ---------------------------------------------------------------------------


def resolve_raw_source(args: DatasetArgs, download_fn: Callable[[], Path]) -> Path:
    """Return a directory containing the raw (pre-transform) dataset contents.

    If ``args.raw_dir`` was supplied (typically via ``--raw-dir``, forwarded by
    the catalog's ``raw_dir`` field — see ``catalog.py``) it is used instead of
    *download_fn*:

    - If it points at a directory, that directory is used directly.
    - If it points at a single archive file (``.zip``/``.tar``/``.tar.gz``), it
      is extracted into a staging directory under ``args.archive_dir`` first.

    This lets a dataset that requires manual or credentialed download (e.g. a
    Kaggle-gated dataset, or one so large that re-downloading it on every run
    is impractical) skip the network fetch step entirely while still running
    through the same transform/export logic as a normal preparation script —
    keeping that logic shared, version-controlled, and testable.

    Otherwise, *download_fn* is called to perform the normal network fetch; it
    should return the local raw-data directory (e.g. after extraction).
    """
    if args.raw_dir is None:
        return download_fn()

    if not args.raw_dir.exists():
        msg = f"--raw-dir {args.raw_dir} does not exist."
        raise FileNotFoundError(msg)

    if args.raw_dir.is_file():
        logger.info("Using pre-fetched raw archive at %s", args.raw_dir)
        return extract_archive(args.raw_dir, args.archive_dir / f"{args.name}_raw_from_raw_dir")

    logger.info("Using pre-fetched raw directory at %s (skipping network download).", args.raw_dir)
    return args.raw_dir


# ---------------------------------------------------------------------------
# Kaggle downloads
# ---------------------------------------------------------------------------


def _has_kaggle_credentials() -> bool:
    """Best-effort check for configured Kaggle API credentials.

    Covers the current Kaggle token flow (``KAGGLE_API_TOKEN`` env var, or a
    ``~/.kaggle/access_token`` file) — see https://www.kaggle.com/docs/api for
    details. Does not validate that the credentials are actually valid, only
    that *something* is configured.
    """
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", "~/.kaggle")).expanduser()
    return (config_dir / "access_token").exists()


def download_kaggle_dataset(dataset_id: str) -> Path:
    """Download a Kaggle dataset via ``kagglehub``.

    Parameters
    ----------
    dataset_id:
        Kaggle dataset identifier, e.g. ``"owner/dataset-slug"``.
    Requires the ``kagglehub`` package to be installed (provided by the
    ``benchmark`` optional dependency extra — see ``pyproject.toml``) and API
    credentials to be configured. Raises a clear, actionable :class:`RuntimeError`
    up front rather than letting the download fail deep inside the dependency,
    and points at ``--raw-dir`` as an alternative for anyone who already has the
    data locally.
    """
    try:
        import kagglehub
    except ImportError as exc:  # pragma: no cover - exercised via unit tests
        msg = (
            "The 'kagglehub' package is not installed. Install benchmark dependencies with "
            "`just venv-benchmark` or `uv sync --extra benchmark` (from library/), or pass --raw-dir with a "
            "manually pre-downloaded copy of this dataset."
        )
        raise RuntimeError(msg) from exc

    if not _has_kaggle_credentials():
        msg = (
            "Kaggle API credentials not found. Configure them via the "
            "KAGGLE_API_TOKEN environment variable, or a ~/.kaggle/access_token file "
            "(see https://www.kaggle.com/docs/api). Alternatively, pass --raw-dir "
            "with a manually pre-downloaded copy of this dataset to skip the download "
            "entirely."
        )
        raise RuntimeError(msg)

    logger.info("Downloading Kaggle dataset %s", dataset_id)
    path = Path(kagglehub.dataset_download(dataset_id))
    if not path.exists():
        msg = f"kagglehub did not produce an expected path for {dataset_id}: {path}"
        raise RuntimeError(msg)
    return path
