# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dataset catalog - loading and script-based provisioning."""

from __future__ import annotations

import logging
import os
import re
import subprocess  # nosec B404 - used to invoke trusted in-repo dataset prep scripts
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Matches an unresolved `${VAR}` or `$VAR` left behind by `os.path.expandvars`
# when the referenced environment variable isn't set.
_UNEXPANDED_VAR_RE = re.compile(r"\$\{[^}]*\}|\$[A-Za-z_][A-Za-z0-9_]*")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetEntry:
    """A single dataset declared in the catalog.

    Exactly one of ``script`` or ``local_path`` must be set:

    - ``script`` — path to a preparation script that downloads/builds the
      dataset (see :func:`provision_dataset`). May be paired with ``raw_dir``
      to skip the script's own network fetch when raw data is already local
      (e.g. a credentialed source such as Kaggle, or a dataset too large to
      conveniently re-download every run).
    - ``local_path`` — a directory that is already fully prepared (placed
      there manually, mounted from a shared/network location, or copied from
      a prior script run). No script is executed; the path is used as-is.
      Supports ``${VAR}``/``~`` expansion so the same catalog entry resolves
      to different locations on different machines/CI runners.
    """

    name: str
    size_tier: str  # tiny | small | medium | large
    script: str | None = None  # path to preparation script (relative to repo root)
    local_path: str | None = None  # pre-existing directory; alternative to `script`
    raw_dir: str | None = None  # optional pre-fetched raw input, forwarded to `script` as --raw-dir
    data_group: str = "all"  # weekly | extended | all - which benchmark lane(s) include this dataset
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if bool(self.script) == bool(self.local_path):
            msg = (
                f"Dataset '{self.name}': exactly one of 'script' or 'local_path' must be set "
                f"(got script={self.script!r}, local_path={self.local_path!r})."
            )
            raise ValueError(msg)
        if self.raw_dir is not None and self.local_path is not None:
            msg = f"Dataset '{self.name}': 'raw_dir' is only valid together with 'script', not 'local_path'."
            raise ValueError(msg)

    @property
    def relative_path(self) -> Path:
        """Return the conventional dataset directory: ``<name>``."""
        return Path(self.name)


@dataclass(frozen=True)
class DatasetCatalog:
    """Parsed representation of ``benchmark_catalog.yaml``."""

    version: int
    datasets: dict[str, DatasetEntry]  # name -> entry

    # -- querying ----------------------------------------------------------

    def all_entries(self) -> list[DatasetEntry]:
        """Return every dataset entry."""
        return list(self.datasets.values())

    def filter(
        self,
        *,
        size_tiers: list[str] | None = None,
        data_groups: list[str] | None = None,
        names: set[str] | None = None,
    ) -> list[DatasetEntry]:
        """Return entries matching **all** supplied filters.

        A dataset with ``data_group: all`` (the default) always matches any
        requested *data_groups* filter; only entries explicitly restricted to
        ``weekly`` or ``extended`` are excluded from the other lane.
        """
        results: list[DatasetEntry] = []
        for entry in self.datasets.values():
            if size_tiers and entry.size_tier not in size_tiers:
                continue
            if data_groups and entry.data_group != "all" and entry.data_group not in data_groups:
                continue
            if names and entry.name not in names:
                continue
            results.append(entry)
        return results

    def get(self, name: str) -> DatasetEntry:
        """Look up a single dataset by name."""
        if name not in self.datasets:
            msg = f"Dataset '{name}' not found in catalog."
            raise KeyError(msg)
        return self.datasets[name]


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


def load_catalog(path: Path) -> DatasetCatalog:
    """Parse ``benchmark_catalog.yaml`` into a :class:`DatasetCatalog`."""
    with Path(path).open() as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    version = raw.get("version", 1)
    datasets: dict[str, DatasetEntry] = {}
    known_keys = {f.name for f in DatasetEntry.__dataclass_fields__.values()}
    for entry_raw in raw.get("datasets", []):
        core = {k: v for k, v in entry_raw.items() if k in known_keys}
        extra = {k: v for k, v in entry_raw.items() if k not in known_keys}
        entry = DatasetEntry(extra=extra, **core)
        datasets[entry.name] = entry
    return DatasetCatalog(version=version, datasets=datasets)


# ---------------------------------------------------------------------------
# Path template expansion (for `local_path` / `raw_dir`)
# ---------------------------------------------------------------------------


def _expand_path_template(template: str, *, dataset_name: str, field_name: str) -> Path:
    """Expand ``${VAR}``/``~`` references in a catalog path template.

    This lets a single catalog entry (e.g. ``local_path: "${GETITUNE_BENCHMARK_EXTERNAL_DATA}/my_ds"``)
    resolve to a different location on each machine/CI runner without editing
    the catalog itself.

    Raises a clear :class:`ValueError` (naming the dataset, field, and
    template) if an environment variable referenced in *template* is not set,
    so a missing external-data mount produces an actionable message instead of
    a confusing "path does not exist" error pointing at a literal ``${...}``
    string.
    """
    expanded = os.path.expandvars(template)
    if _UNEXPANDED_VAR_RE.search(expanded):
        msg = (
            f"Dataset '{dataset_name}': could not resolve an environment variable in "
            f"{field_name}='{template}' (result: '{expanded}'). Set the referenced "
            "environment variable(s) and try again."
        )
        raise ValueError(msg)
    return Path(expanded).expanduser()


# ---------------------------------------------------------------------------
# Script execution
# ---------------------------------------------------------------------------


def _resolve_script_path(script: str) -> Path:
    """Resolve a script path relative to the repository root.

    The *script* field in the catalog is relative to the repo root.
    We walk up from the ``catalog.py`` source file to find ``library/``.
    """
    # repo root is 4 levels up (src/getitune/benchmark/catalog.py -> library/)
    library_root = Path(__file__).resolve().parents[3]
    return library_root / script


def _run_script(script_path: Path, data_root: Path, name: str, *, raw_dir: Path | None = None) -> None:
    """Execute a dataset preparation script and stream its output in real time.

    The child's stdout/stderr are merged and forwarded line-by-line to the
    logger so that long-running download/extract steps are visible to the
    user.

    When *raw_dir* is given, it is forwarded as ``--raw-dir`` so the script can
    skip its own network download (see ``dataset_helpers.resolve_raw_source``).

    The call blocks until the child exits; a non-zero exit code raises
    :class:`RuntimeError`.
    """
    logger.info("Running preparation script: %s (dataset=%s)", script_path, name)

    cmd = [
        sys.executable,
        "-u",
        str(script_path),
        "--output-dir",
        str(data_root),
        "--name",
        name,
    ]
    if raw_dir is not None:
        cmd.extend(["--raw-dir", str(raw_dir)])

    # Merge stderr into stdout so log ordering matches the child's own
    # write order (and so a single reader loop is sufficient).
    proc = subprocess.Popen(  # noqa: S603
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # line-buffered
    )

    if proc.stdout is None:
        msg = "subprocess.PIPE guarantee violated for benchmark command output."
        raise RuntimeError(msg)
    try:
        for raw_line in proc.stdout:
            line = raw_line.rstrip()
            if line:
                logger.info("  [%s] %s", name, line)
    finally:
        # Always wait for the child so we never return before it finishes.
        returncode = proc.wait()

    if returncode != 0:
        msg = f"Preparation script for dataset '{name}' failed with exit code {returncode}.\nScript: {script_path}"
        raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Provisioning
# ---------------------------------------------------------------------------


def provision_local_dataset(entry: DatasetEntry) -> Path:
    """Resolve a catalog entry's ``local_path`` to a verified, existing directory.

    No script is executed and no ``.ready`` sentinel is written: the directory
    is assumed to be externally managed (e.g. a shared network mount, or a
    directory a person populated manually after a credentialed download), so
    existence is simply re-checked on every call rather than cached.
    """
    if entry.local_path is None:
        msg = f"Dataset '{entry.name}' has no 'local_path' to provision."
        raise ValueError(msg)

    path = _expand_path_template(entry.local_path, dataset_name=entry.name, field_name="local_path")

    if not path.exists():
        msg = (
            f"Dataset '{entry.name}': local_path '{entry.local_path}' resolved to '{path}', which "
            "does not exist. Place the prepared dataset there — see the benchmark README's "
            "'Datasets requiring credentials or manual placement' section — and try again."
        )
        raise FileNotFoundError(msg)
    if not path.is_dir():
        msg = f"Dataset '{entry.name}': local_path '{path}' exists but is not a directory."
        raise NotADirectoryError(msg)

    logger.info("Dataset '%s' provided via local_path at %s (no preparation script run).", entry.name, path)
    return path


def _resolve_raw_dir(entry: DatasetEntry) -> Path | None:
    """Best-effort resolution of a catalog entry's optional ``raw_dir``.

    Unlike ``local_path`` (required — a hard error if unresolvable, since
    there is no fallback), ``raw_dir`` is an optional accelerant: if its
    template references an unset environment variable, or the resolved path
    doesn't exist, we log a warning and return ``None`` so the caller falls
    back to the preparation script's normal (e.g. network / credentialed)
    download path instead of failing outright.
    """
    if entry.raw_dir is None:
        return None

    try:
        candidate = _expand_path_template(entry.raw_dir, dataset_name=entry.name, field_name="raw_dir")
    except ValueError as exc:
        logger.warning(
            "Dataset '%s': raw_dir could not be resolved (%s); falling back to the script's "
            "normal (network) download path.",
            entry.name,
            exc,
        )
        return None

    if not candidate.exists():
        logger.warning(
            "Dataset '%s': raw_dir configured at '%s' but not found; falling back to the "
            "script's normal (network) download path.",
            entry.name,
            candidate,
        )
        return None

    logger.info("Dataset '%s': forwarding raw_dir '%s' to the preparation script.", entry.name, candidate)
    return candidate


def provision_dataset(entry: DatasetEntry, data_root: Path) -> Path:
    """Ensure a single dataset is prepared and ready.

    Datasets declared with ``local_path`` are resolved directly (see
    :func:`provision_local_dataset`) — no script runs for them.

    Otherwise, a ``.ready`` sentinel file is written inside the dataset
    directory once the preparation script finishes successfully. On
    subsequent runs the sentinel is what we check — a stale, half-populated
    directory left behind by a crashed prep run is therefore treated as "not
    ready" and the script is re-executed.

    Returns the path to the prepared dataset directory.
    """
    if entry.local_path is not None:
        return provision_local_dataset(entry)

    dataset_dir = data_root / entry.relative_path
    ready_marker = dataset_dir / ".ready"

    if ready_marker.exists():
        logger.info("Dataset '%s' already exists, skipping.", entry.name)
        return dataset_dir

    if dataset_dir.exists():
        logger.warning(
            "Dataset '%s' directory exists but is missing the readiness marker; re-running prep.",
            entry.name,
        )

    if entry.script is None:
        msg = f"Dataset '{entry.name}' has no preparation script defined."
        raise RuntimeError(msg)
    script_path = _resolve_script_path(entry.script)

    if not script_path.exists():
        msg = f"Preparation script not found for dataset '{entry.name}': {script_path}"
        raise FileNotFoundError(msg)

    raw_dir = _resolve_raw_dir(entry)

    _run_script(script_path, data_root, entry.name, raw_dir=raw_dir)

    if not dataset_dir.exists():
        msg = f"Preparation script for '{entry.name}' did not create expected directory: {dataset_dir}"
        raise RuntimeError(msg)

    # Write the readiness sentinel only after a clean run so an interrupted
    # prep is never mistaken for a successful one on a later resume.
    ready_marker.touch()

    return dataset_dir


def provision_datasets(
    catalog: DatasetCatalog,
    data_root: Path,
    *,
    entries: list[DatasetEntry] | None = None,
) -> dict[str, Path]:
    """Run preparation scripts for all datasets (or a filtered subset).

    Each dataset is provisioned independently: if one entry's script fails
    (e.g. a transient network error, or missing Kaggle credentials), the
    error is logged and that dataset is omitted from the result rather than
    aborting the whole batch. Callers (e.g. :class:`~getitune.benchmark.runner.BenchmarkRunner`)
    already treat a missing dataset name as "skip the experiments that need it".

    Returns a mapping ``{dataset_name: prepared_path}`` containing only the
    datasets that were provisioned successfully.
    """
    targets = entries if entries is not None else catalog.all_entries()
    result: dict[str, Path] = {}
    for entry in targets:
        try:
            result[entry.name] = provision_dataset(entry, data_root)
        except Exception:  # noqa: PERF203 - isolating failures per dataset is the whole point here
            logger.exception("Failed to provision dataset '%s'; skipping.", entry.name)
    return result
