# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for getitune.benchmark.catalog."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from getitune.benchmark.catalog import (
    DatasetCatalog,
    DatasetEntry,
    _expand_path_template,
    load_catalog,
    provision_dataset,
    provision_datasets,
    provision_local_dataset,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def catalog_yaml(tmp_path: Path) -> Path:
    """Write a minimal catalog YAML and return its path."""
    content = textwrap.dedent("""\
        version: 1
        datasets:
          - name: ds_weekly
            script: "scripts/benchmark_datasets/prepare_ds_weekly.py"
            size_tier: medium
            data_group: weekly
          - name: ds_tiny
            script: "scripts/benchmark_datasets/prepare_ds_tiny.py"
            size_tier: tiny
          - name: ds_small
            script: "scripts/benchmark_datasets/prepare_ds_small.py"
            size_tier: small
          - name: cls_tiny
            script: "scripts/benchmark_datasets/prepare_cls_tiny.py"
            size_tier: tiny
    """)
    p = tmp_path / "catalog.yaml"
    p.write_text(content)
    return p


@pytest.fixture
def catalog(catalog_yaml: Path) -> DatasetCatalog:
    return load_catalog(catalog_yaml)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class TestLoadCatalog:
    def test_loads_version(self, catalog: DatasetCatalog) -> None:
        assert catalog.version == 1

    def test_parses_all_entries(self, catalog: DatasetCatalog) -> None:
        assert len(catalog.all_entries()) == 4

    def test_dataset_keys(self, catalog: DatasetCatalog) -> None:
        assert set(catalog.datasets.keys()) == {"ds_weekly", "ds_tiny", "ds_small", "cls_tiny"}

    def test_entry_fields(self, catalog: DatasetCatalog) -> None:
        entry = catalog.get("ds_tiny")
        assert entry.name == "ds_tiny"
        assert entry.script == "scripts/benchmark_datasets/prepare_ds_tiny.py"
        assert entry.size_tier == "tiny"

    def test_data_group_defaults_to_all(self, catalog: DatasetCatalog) -> None:
        assert catalog.get("ds_tiny").data_group == "all"

    def test_data_group_explicit_value(self, catalog: DatasetCatalog) -> None:
        assert catalog.get("ds_weekly").data_group == "weekly"

    def test_get_unknown_raises(self, catalog: DatasetCatalog) -> None:
        with pytest.raises(KeyError, match="not_real"):
            catalog.get("not_real")


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


class TestCatalogFilter:
    def test_filter_by_size_tier(self, catalog: DatasetCatalog) -> None:
        results = catalog.filter(size_tiers=["tiny"])
        assert len(results) == 2
        names = {e.name for e in results}
        assert names == {"ds_tiny", "cls_tiny"}

    def test_filter_by_name(self, catalog: DatasetCatalog) -> None:
        results = catalog.filter(names={"ds_small"})
        assert len(results) == 1
        assert results[0].name == "ds_small"

    def test_combined_filters(self, catalog: DatasetCatalog) -> None:
        results = catalog.filter(size_tiers=["tiny"], names={"ds_tiny"})
        assert len(results) == 1
        assert results[0].name == "ds_tiny"

    def test_no_match_returns_empty(self, catalog: DatasetCatalog) -> None:
        assert catalog.filter(size_tiers=["large"]) == []

    def test_filter_by_data_group_weekly_matches_explicit_and_default(self, catalog: DatasetCatalog) -> None:
        results = catalog.filter(data_groups=["weekly"])
        names = {e.name for e in results}
        # ds_weekly is explicitly "weekly"; the rest default to "all" and always match.
        assert names == {"ds_weekly", "ds_tiny", "ds_small", "cls_tiny"}

    def test_filter_by_data_group_extended_excludes_weekly_only(self, catalog: DatasetCatalog) -> None:
        results = catalog.filter(data_groups=["extended"])
        names = {e.name for e in results}
        assert names == {"ds_tiny", "ds_small", "cls_tiny"}


# ---------------------------------------------------------------------------
# Relative path
# ---------------------------------------------------------------------------


class TestDatasetEntry:
    def test_relative_path(self) -> None:
        entry = DatasetEntry(
            name="my_ds",
            script="scripts/prepare.py",
            size_tier="tiny",
        )
        assert entry.relative_path == Path("my_ds")


# ---------------------------------------------------------------------------
# DatasetEntry validation (script / local_path / raw_dir)
# ---------------------------------------------------------------------------


class TestDatasetEntryValidation:
    def test_script_only_is_valid(self) -> None:
        entry = DatasetEntry(name="ds", script="scripts/prepare.py", size_tier="tiny")
        assert entry.script == "scripts/prepare.py"
        assert entry.local_path is None

    def test_local_path_only_is_valid(self) -> None:
        entry = DatasetEntry(name="ds", local_path="/mnt/data/ds", size_tier="tiny")
        assert entry.local_path == "/mnt/data/ds"
        assert entry.script is None

    def test_neither_script_nor_local_path_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly one of 'script' or 'local_path'"):
            DatasetEntry(name="ds", size_tier="tiny")

    def test_both_script_and_local_path_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly one of 'script' or 'local_path'"):
            DatasetEntry(name="ds", script="scripts/prepare.py", local_path="/mnt/data/ds", size_tier="tiny")

    def test_raw_dir_with_script_is_valid(self) -> None:
        entry = DatasetEntry(name="ds", script="scripts/prepare.py", raw_dir="/mnt/raw/ds", size_tier="tiny")
        assert entry.raw_dir == "/mnt/raw/ds"

    def test_raw_dir_with_local_path_raises(self) -> None:
        with pytest.raises(ValueError, match="'raw_dir' is only valid together with 'script'"):
            DatasetEntry(name="ds", local_path="/mnt/data/ds", raw_dir="/mnt/raw/ds", size_tier="tiny")

    def test_script_with_raw_dir_is_valid(self) -> None:
        entry = DatasetEntry(name="ds", script="scripts/prepare.py", raw_dir="/mnt/raw/ds", size_tier="tiny")
        assert entry.script == "scripts/prepare.py"
        assert entry.raw_dir == "/mnt/raw/ds"


# ---------------------------------------------------------------------------
# Path template expansion (local_path / raw_dir)
# ---------------------------------------------------------------------------


class TestExpandPathTemplate:
    def test_expands_set_env_var(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("MY_TEST_ROOT", str(tmp_path))
        result = _expand_path_template("${MY_TEST_ROOT}/sub", dataset_name="ds", field_name="local_path")
        assert result == tmp_path / "sub"

    def test_expands_user_home(self) -> None:
        result = _expand_path_template("~/some_dir", dataset_name="ds", field_name="local_path")
        assert "~" not in str(result)
        assert str(result).endswith("some_dir")

    def test_unset_env_var_raises_clear_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEFINITELY_UNSET_VAR_XYZ", raising=False)
        with pytest.raises(ValueError, match="DEFINITELY_UNSET_VAR_XYZ"):
            _expand_path_template("${DEFINITELY_UNSET_VAR_XYZ}/sub", dataset_name="my_ds", field_name="local_path")

    def test_error_message_names_dataset_and_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("UNSET_VAR_ABC", raising=False)
        with pytest.raises(ValueError, match=r"Dataset 'my_ds'.*raw_dir"):
            _expand_path_template("${UNSET_VAR_ABC}/sub", dataset_name="my_ds", field_name="raw_dir")

    def test_plain_path_passes_through(self) -> None:
        result = _expand_path_template("/mnt/data/ds", dataset_name="ds", field_name="local_path")
        assert result == Path("/mnt/data/ds")


# ---------------------------------------------------------------------------
# Helper: create a simple preparation script
# ---------------------------------------------------------------------------


def _make_prep_script(script_path: Path, file_content: str = "world") -> None:
    """Create a minimal preparation script that creates the dataset directory."""
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(
        textwrap.dedent(f"""\
        import argparse
        from pathlib import Path

        parser = argparse.ArgumentParser()
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--name", type=str, required=True)
        args = parser.parse_args()

        dest = args.output_dir / args.name
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "hello.txt").write_text("{file_content}")
    """)
    )


class TestProvisionDataset:
    def test_skips_when_directory_exists(self, tmp_path: Path) -> None:
        """If the dataset is marked ready, the script should not run."""
        data_root = tmp_path / "data"
        ds_dir = data_root / "cached_ds"
        ds_dir.mkdir(parents=True)
        # The readiness sentinel is what marks a dataset as already prepared;
        # an empty directory alone is treated as a stale prep run.
        (ds_dir / ".ready").touch()

        entry = DatasetEntry(
            name="cached_ds",
            script="scripts/prepare.py",
            size_tier="tiny",
        )
        # No need to mock _resolve_script_path — it should never be called
        result = provision_dataset(entry, data_root)
        assert result == ds_dir

    def test_run_script_and_provision(self, tmp_path: Path) -> None:
        """Script should be run and dataset dir created."""
        data_root = tmp_path / "data"

        script_path = tmp_path / "scripts" / "prepare.py"
        _make_prep_script(script_path)

        entry = DatasetEntry(
            name="test_ds",
            script="scripts/prepare.py",
            size_tier="tiny",
        )

        with patch("getitune.benchmark.catalog._resolve_script_path", return_value=script_path):
            result = provision_dataset(entry, data_root)

        assert result.exists()
        assert (result / "hello.txt").exists()
        assert (result / "hello.txt").read_text() == "world"

    def test_script_not_found_raises(self, tmp_path: Path) -> None:
        """Missing preparation script must raise FileNotFoundError."""
        data_root = tmp_path / "data"
        entry = DatasetEntry(
            name="missing",
            script="scripts/does_not_exist.py",
            size_tier="tiny",
        )

        with patch(
            "getitune.benchmark.catalog._resolve_script_path",
            return_value=tmp_path / "scripts" / "does_not_exist.py",
        ), pytest.raises(FileNotFoundError, match="Preparation script not found"):
            provision_dataset(entry, data_root)

    def test_script_failure_raises(self, tmp_path: Path) -> None:
        """A script that exits non-zero must raise RuntimeError."""
        data_root = tmp_path / "data"
        script_path = tmp_path / "scripts" / "bad.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("import sys; sys.exit(1)\n")

        entry = DatasetEntry(
            name="bad_ds",
            script="scripts/bad.py",
            size_tier="tiny",
        )

        with patch("getitune.benchmark.catalog._resolve_script_path", return_value=script_path), pytest.raises(
            RuntimeError, match="failed with exit code"
        ):
            provision_dataset(entry, data_root)


# ---------------------------------------------------------------------------
# Local (pre-prepared) datasets
# ---------------------------------------------------------------------------


class TestProvisionLocalDataset:
    def test_resolves_existing_directory(self, tmp_path: Path) -> None:
        ds_dir = tmp_path / "prepared_ds"
        ds_dir.mkdir()
        entry = DatasetEntry(name="ds", local_path=str(ds_dir), size_tier="tiny")

        result = provision_local_dataset(entry)

        assert result == ds_dir

    def test_expands_env_var(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        external_root = tmp_path / "external"
        ds_dir = external_root / "my_ds"
        ds_dir.mkdir(parents=True)
        monkeypatch.setenv("GETITUNE_BENCHMARK_EXTERNAL_DATA", str(external_root))

        entry = DatasetEntry(
            name="my_ds",
            local_path="${GETITUNE_BENCHMARK_EXTERNAL_DATA}/my_ds",
            size_tier="tiny",
        )

        result = provision_local_dataset(entry)

        assert result == ds_dir

    def test_missing_directory_raises_file_not_found(self, tmp_path: Path) -> None:
        entry = DatasetEntry(name="ds", local_path=str(tmp_path / "does_not_exist"), size_tier="tiny")

        with pytest.raises(FileNotFoundError, match="does not exist"):
            provision_local_dataset(entry)

    def test_unset_env_var_raises_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SOME_UNSET_BENCHMARK_VAR", raising=False)
        entry = DatasetEntry(name="ds", local_path="${SOME_UNSET_BENCHMARK_VAR}/ds", size_tier="tiny")

        with pytest.raises(ValueError, match="SOME_UNSET_BENCHMARK_VAR"):
            provision_local_dataset(entry)

    def test_path_is_a_file_raises_not_a_directory(self, tmp_path: Path) -> None:
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("oops")
        entry = DatasetEntry(name="ds", local_path=str(file_path), size_tier="tiny")

        with pytest.raises(NotADirectoryError, match="not a directory"):
            provision_local_dataset(entry)

    def test_provision_dataset_dispatches_to_local_path(self, tmp_path: Path) -> None:
        """provision_dataset() should route local_path entries without touching data_root."""
        ds_dir = tmp_path / "prepared_ds"
        ds_dir.mkdir()
        entry = DatasetEntry(name="ds", local_path=str(ds_dir), size_tier="tiny")

        # data_root is unrelated/nonexistent — proving it's never consulted for local_path entries.
        result = provision_dataset(entry, tmp_path / "unrelated_data_root")

        assert result == ds_dir


# ---------------------------------------------------------------------------
# raw_dir forwarding (script-based datasets with a pre-fetched raw input)
# ---------------------------------------------------------------------------


def _make_argv_recording_script(script_path: Path) -> None:
    """Create a preparation script that records its received CLI args as JSON.

    Also creates the expected ``<output-dir>/<name>`` directory so provisioning
    considers the run successful, letting tests inspect ``received_args.json``
    inside the dataset directory afterwards to assert on forwarded flags.
    """
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(
        textwrap.dedent("""\
        import argparse
        import json
        from pathlib import Path

        parser = argparse.ArgumentParser()
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--name", type=str, required=True)
        parser.add_argument("--raw-dir", type=Path, default=None)
        args = parser.parse_args()

        dest = args.output_dir / args.name
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "received_args.json").write_text(
            json.dumps({"raw_dir": str(args.raw_dir) if args.raw_dir else None})
        )
    """)
    )


class TestProvisionDatasetRawDir:
    def test_forwards_existing_raw_dir(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        script_path = tmp_path / "scripts" / "prepare.py"
        _make_argv_recording_script(script_path)

        raw_source = tmp_path / "raw_source"
        raw_source.mkdir()

        entry = DatasetEntry(name="ds", script="scripts/prepare.py", raw_dir=str(raw_source), size_tier="tiny")

        with patch("getitune.benchmark.catalog._resolve_script_path", return_value=script_path):
            result = provision_dataset(entry, data_root)

        received = json.loads((result / "received_args.json").read_text())
        assert received["raw_dir"] == str(raw_source)

    def test_falls_back_when_raw_dir_missing(self, tmp_path: Path) -> None:
        """A raw_dir that doesn't exist on disk should not be forwarded; script runs normally."""
        data_root = tmp_path / "data"
        script_path = tmp_path / "scripts" / "prepare.py"
        _make_argv_recording_script(script_path)

        entry = DatasetEntry(
            name="ds",
            script="scripts/prepare.py",
            raw_dir=str(tmp_path / "nonexistent_raw"),
            size_tier="tiny",
        )

        with patch("getitune.benchmark.catalog._resolve_script_path", return_value=script_path):
            result = provision_dataset(entry, data_root)

        received = json.loads((result / "received_args.json").read_text())
        assert received["raw_dir"] is None

    def test_falls_back_when_raw_dir_env_var_unset(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """An unresolvable ${VAR} in raw_dir must not abort provisioning (unlike local_path)."""
        monkeypatch.delenv("UNSET_RAW_DIR_VAR", raising=False)
        data_root = tmp_path / "data"
        script_path = tmp_path / "scripts" / "prepare.py"
        _make_argv_recording_script(script_path)

        entry = DatasetEntry(
            name="ds",
            script="scripts/prepare.py",
            raw_dir="${UNSET_RAW_DIR_VAR}/raw",
            size_tier="tiny",
        )

        with patch("getitune.benchmark.catalog._resolve_script_path", return_value=script_path):
            result = provision_dataset(entry, data_root)  # must not raise

        received = json.loads((result / "received_args.json").read_text())
        assert received["raw_dir"] is None

    def test_no_raw_dir_configured(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        script_path = tmp_path / "scripts" / "prepare.py"
        _make_argv_recording_script(script_path)

        entry = DatasetEntry(name="ds", script="scripts/prepare.py", size_tier="tiny")

        with patch("getitune.benchmark.catalog._resolve_script_path", return_value=script_path):
            result = provision_dataset(entry, data_root)

        received = json.loads((result / "received_args.json").read_text())
        assert received["raw_dir"] is None


# ---------------------------------------------------------------------------
# Extra fields & edge cases
# ---------------------------------------------------------------------------


class TestLoadCatalogExtras:
    def test_unknown_keys_go_to_extra(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            version: 1
            datasets:
              - name: ds_custom
                script: "scripts/prepare.py"
                size_tier: tiny
                custom_field: 42
                another_field: hello
        """)
        p = tmp_path / "catalog.yaml"
        p.write_text(content)
        catalog = load_catalog(p)
        entry = catalog.get("ds_custom")
        assert entry.extra["custom_field"] == 42
        assert entry.extra["another_field"] == "hello"

    def test_default_version(self, tmp_path: Path) -> None:
        """Catalog without explicit version should default to 1."""
        content = textwrap.dedent("""\
            datasets:
              - name: ds
                script: "scripts/prepare.py"
                size_tier: tiny
        """)
        p = tmp_path / "catalog.yaml"
        p.write_text(content)
        catalog = load_catalog(p)
        assert catalog.version == 1

    def test_empty_datasets_section(self, tmp_path: Path) -> None:
        content = "version: 1\ndatasets: []\n"
        p = tmp_path / "catalog.yaml"
        p.write_text(content)
        catalog = load_catalog(p)
        assert catalog.all_entries() == []

    def test_filter_no_args_returns_all(self, catalog: DatasetCatalog) -> None:
        """Calling filter with no arguments returns everything."""
        results = catalog.filter()
        assert len(results) == 4


# ---------------------------------------------------------------------------
# Provision multiple datasets
# ---------------------------------------------------------------------------


class TestProvisionDatasets:
    def test_provisions_all_entries(self, tmp_path: Path) -> None:
        script_a = tmp_path / "scripts" / "prepare_a.py"
        script_b = tmp_path / "scripts" / "prepare_b.py"
        _make_prep_script(script_a, file_content="content_a")
        _make_prep_script(script_b, file_content="content_b")

        entry_a = DatasetEntry(name="a", script="scripts/prepare_a.py", size_tier="tiny")
        entry_b = DatasetEntry(name="b", script="scripts/prepare_b.py", size_tier="tiny")
        catalog = DatasetCatalog(version=1, datasets={"a": entry_a, "b": entry_b})

        data_root = tmp_path / "data"

        def resolve(script: str) -> Path:
            return tmp_path / script

        with patch("getitune.benchmark.catalog._resolve_script_path", side_effect=resolve):
            result = provision_datasets(catalog, data_root)

        assert set(result.keys()) == {"a", "b"}
        assert (result["a"] / "hello.txt").exists()
        assert (result["b"] / "hello.txt").exists()

    def test_provisions_subset(self, tmp_path: Path) -> None:
        script_a = tmp_path / "scripts" / "prepare_a.py"
        _make_prep_script(script_a, file_content="data")

        entry_a = DatasetEntry(name="a", script="scripts/prepare_a.py", size_tier="tiny")
        entry_b = DatasetEntry(name="b", script="scripts/prepare_b.py", size_tier="tiny")
        catalog = DatasetCatalog(version=1, datasets={"a": entry_a, "b": entry_b})

        data_root = tmp_path / "data"

        def resolve(script: str) -> Path:
            return tmp_path / script

        with patch("getitune.benchmark.catalog._resolve_script_path", side_effect=resolve):
            result = provision_datasets(catalog, data_root, entries=[entry_a])

        assert "a" in result
        assert "b" not in result


# ---------------------------------------------------------------------------
# Resilience: one dataset's failure must not abort the whole batch
# ---------------------------------------------------------------------------


class TestProvisionDatasetsResilience:
    def test_one_failing_script_does_not_abort_others(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """A single failing dataset must be logged & skipped, not raised."""
        good_script = tmp_path / "scripts" / "prepare_good.py"
        _make_prep_script(good_script, file_content="ok")
        bad_script = tmp_path / "scripts" / "prepare_bad.py"
        bad_script.parent.mkdir(parents=True, exist_ok=True)
        bad_script.write_text("import sys; sys.exit(1)\n")

        entry_good = DatasetEntry(name="good", script="scripts/prepare_good.py", size_tier="tiny")
        entry_bad = DatasetEntry(name="bad", script="scripts/prepare_bad.py", size_tier="tiny")
        catalog = DatasetCatalog(version=1, datasets={"good": entry_good, "bad": entry_bad})

        data_root = tmp_path / "data"

        def resolve(script: str) -> Path:
            return tmp_path / script

        with (
            patch("getitune.benchmark.catalog._resolve_script_path", side_effect=resolve),
            caplog.at_level("ERROR", logger="getitune.benchmark.catalog"),
        ):
            result = provision_datasets(catalog, data_root)

        # The good dataset still gets provisioned...
        assert "good" in result
        assert (result["good"] / "hello.txt").exists()
        # ...while the bad one is cleanly omitted rather than raising.
        assert "bad" not in result
        assert any("bad" in record.message for record in caplog.records)

    def test_missing_local_path_does_not_abort_others(self, tmp_path: Path) -> None:
        good_script = tmp_path / "scripts" / "prepare_good.py"
        _make_prep_script(good_script, file_content="ok")

        entry_good = DatasetEntry(name="good", script="scripts/prepare_good.py", size_tier="tiny")
        entry_local = DatasetEntry(name="unreachable", local_path=str(tmp_path / "nope"), size_tier="tiny")
        catalog = DatasetCatalog(version=1, datasets={"good": entry_good, "unreachable": entry_local})

        data_root = tmp_path / "data"

        def resolve(script: str) -> Path:
            return tmp_path / script

        with patch("getitune.benchmark.catalog._resolve_script_path", side_effect=resolve):
            result = provision_datasets(catalog, data_root)

        assert "good" in result
        assert "unreachable" not in result

    def test_all_failing_returns_empty_without_raising(self, tmp_path: Path) -> None:
        entry = DatasetEntry(name="unreachable", local_path=str(tmp_path / "nope"), size_tier="tiny")
        catalog = DatasetCatalog(version=1, datasets={"unreachable": entry})

        result = provision_datasets(catalog, tmp_path / "data")

        assert result == {}
