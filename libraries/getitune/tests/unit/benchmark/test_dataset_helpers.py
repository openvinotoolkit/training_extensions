# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for getitune.benchmark.dataset_helpers."""

from __future__ import annotations

import io
import sys
import tarfile
import zipfile
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

from getitune.benchmark.dataset_helpers import (
    DatasetArgs,
    _has_kaggle_credentials,
    download,
    download_kaggle_dataset,
    extract_archive,
    parse_args,
    resolve_raw_source,
)

# ---------------------------------------------------------------------------
# DatasetArgs
# ---------------------------------------------------------------------------


class TestDatasetArgs:
    def test_dest_property(self) -> None:
        args = DatasetArgs(output_dir=Path("/data"), name="my_ds")
        assert args.dest == Path("/data/my_ds")

    def test_archive_dir_property(self) -> None:
        args = DatasetArgs(output_dir=Path("/data"), name="my_ds")
        assert args.archive_dir == Path("/data/.archives")

    def test_raw_dir_defaults_to_none(self) -> None:
        args = DatasetArgs(output_dir=Path("/data"), name="my_ds")
        assert args.raw_dir is None


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_parses_required_args(self, tmp_path: Path) -> None:
        data_dir = str(tmp_path / "data")
        with patch("sys.argv", ["prog", "--output-dir", data_dir, "--name", "ds_a"]):
            args = parse_args()
        assert args.output_dir == Path(data_dir)
        assert args.name == "ds_a"
        assert args.dest == Path(data_dir) / "ds_a"
        assert args.raw_dir is None

    def test_missing_args_exits(self) -> None:
        with patch("sys.argv", ["prog"]), pytest.raises(SystemExit):
            parse_args()

    def test_parses_optional_raw_dir(self, tmp_path: Path) -> None:
        data_dir = str(tmp_path / "data")
        raw_dir = str(tmp_path / "raw")
        with patch(
            "sys.argv",
            ["prog", "--output-dir", data_dir, "--name", "ds_a", "--raw-dir", raw_dir],
        ):
            args = parse_args()
        assert args.raw_dir == Path(raw_dir)


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------


class TestDownload:
    def test_downloads_to_dest(self, tmp_path: Path) -> None:
        dest_dir = tmp_path / "archives"

        # Mock urlretrieve to just create a file
        def fake_urlretrieve(url: str, dest: str | Path) -> None:
            Path(dest).write_text("fake archive content")

        with patch("getitune.benchmark.dataset_helpers.urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            result = download("https://example.com/data.tar.gz", dest_dir)

        assert result == dest_dir / "data.tar.gz"
        assert result.read_text() == "fake archive content"

    def test_custom_filename(self, tmp_path: Path) -> None:
        dest_dir = tmp_path / "archives"

        def fake_urlretrieve(url: str, dest: str | Path) -> None:
            Path(dest).write_text("content")

        with patch("getitune.benchmark.dataset_helpers.urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            result = download("https://example.com/v2/data.tar.gz", dest_dir, filename="custom.tar.gz")

        assert result == dest_dir / "custom.tar.gz"

    def test_creates_dest_dir(self, tmp_path: Path) -> None:
        dest_dir = tmp_path / "nested" / "deep" / "dir"
        assert not dest_dir.exists()

        def fake_urlretrieve(url: str, dest: str | Path) -> None:
            Path(dest).write_text("content")

        with patch("getitune.benchmark.dataset_helpers.urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            download("https://example.com/a.zip", dest_dir)

        assert dest_dir.exists()


# ---------------------------------------------------------------------------
# extract_archive
# ---------------------------------------------------------------------------


def _make_tar_gz(path: Path, files: dict[str, str]) -> None:
    """Create a .tar.gz archive with the given files."""
    with tarfile.open(path, "w:gz") as tf:
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def _make_zip(path: Path, files: dict[str, str]) -> None:
    """Create a .zip archive with the given files."""
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)


class TestExtractArchive:
    def test_extract_tar_gz(self, tmp_path: Path) -> None:
        archive = tmp_path / "data.tar.gz"
        _make_tar_gz(archive, {"file.txt": "hello"})

        dest = tmp_path / "output"
        result = extract_archive(archive, dest)

        assert result == dest
        assert (dest / "file.txt").read_text() == "hello"

    def test_extract_zip(self, tmp_path: Path) -> None:
        archive = tmp_path / "data.zip"
        _make_zip(archive, {"readme.md": "# Readme"})

        dest = tmp_path / "output"
        result = extract_archive(archive, dest)

        assert result == dest
        assert (dest / "readme.md").read_text() == "# Readme"

    def test_clean_dest_removes_existing(self, tmp_path: Path) -> None:
        dest = tmp_path / "output"
        dest.mkdir()
        (dest / "stale.txt").write_text("old content")

        archive = tmp_path / "data.zip"
        _make_zip(archive, {"fresh.txt": "new"})

        extract_archive(archive, dest, clean_dest=True)

        assert (dest / "fresh.txt").exists()
        assert not (dest / "stale.txt").exists()

    def test_no_clean_dest_preserves_existing(self, tmp_path: Path) -> None:
        dest = tmp_path / "output"
        dest.mkdir()
        (dest / "keep.txt").write_text("kept")

        archive = tmp_path / "data.zip"
        _make_zip(archive, {"added.txt": "new"})

        extract_archive(archive, dest, clean_dest=False)

        assert (dest / "added.txt").exists()
        assert (dest / "keep.txt").exists()

    def test_unsupported_format_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.dat"
        bad_file.write_text("not an archive")
        with pytest.raises(ValueError, match="Unsupported archive format"):
            extract_archive(bad_file, tmp_path / "out")


# ---------------------------------------------------------------------------
# resolve_raw_source
# ---------------------------------------------------------------------------


class TestResolveRawSource:
    def test_calls_download_fn_when_raw_dir_is_none(self, tmp_path: Path) -> None:
        args = DatasetArgs(output_dir=tmp_path / "data", name="ds")
        sentinel = tmp_path / "downloaded"
        download_fn = lambda: sentinel

        result = resolve_raw_source(args, download_fn)

        assert result == sentinel

    def test_uses_raw_dir_directory_directly(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "image.png").write_text("fake")
        args = DatasetArgs(output_dir=tmp_path / "data", name="ds", raw_dir=raw_dir)

        called = False

        def download_fn() -> Path:
            nonlocal called
            called = True
            return tmp_path / "should_not_be_used"

        result = resolve_raw_source(args, download_fn)

        assert result == raw_dir
        assert not called

    def test_extracts_raw_dir_archive_file(self, tmp_path: Path) -> None:
        archive = tmp_path / "raw.zip"
        _make_zip(archive, {"image.png": "fake-bytes"})
        args = DatasetArgs(output_dir=tmp_path / "data", name="ds", raw_dir=archive)

        result = resolve_raw_source(args, lambda: (_ for _ in ()).throw(AssertionError("should not download")))

        assert result == args.archive_dir / "ds_raw_from_raw_dir"
        assert (result / "image.png").read_text() == "fake-bytes"

    def test_missing_raw_dir_raises_file_not_found(self, tmp_path: Path) -> None:
        args = DatasetArgs(output_dir=tmp_path / "data", name="ds", raw_dir=tmp_path / "does_not_exist")

        with pytest.raises(FileNotFoundError, match="does not exist"):
            resolve_raw_source(args, lambda: tmp_path / "unused")


# ---------------------------------------------------------------------------
# Kaggle credentials
# ---------------------------------------------------------------------------


class TestHasKaggleCredentials:
    def _clear_ambient_credentials(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Isolate from the real environment/host `~/.kaggle` (which may exist)."""
        monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
        monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path / "empty_kaggle_config"))

    def test_no_credentials_returns_false(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self._clear_ambient_credentials(monkeypatch, tmp_path)
        assert _has_kaggle_credentials() is False

    def test_api_token_env_var(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self._clear_ambient_credentials(monkeypatch, tmp_path)
        monkeypatch.setenv("KAGGLE_API_TOKEN", "token-value")
        assert _has_kaggle_credentials() is True

    def test_access_token_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self._clear_ambient_credentials(monkeypatch, tmp_path)
        config_dir = tmp_path / "kaggle_config_with_token"
        config_dir.mkdir()
        (config_dir / "access_token").write_text("token-value")
        monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(config_dir))
        assert _has_kaggle_credentials() is True


# ---------------------------------------------------------------------------
# download_kaggle_dataset
# ---------------------------------------------------------------------------


class TestDownloadKaggleDataset:
    def test_raises_when_cli_missing(self, tmp_path: Path) -> None:
        with (
            patch.dict(sys.modules, {"kagglehub": None}),
            pytest.raises(RuntimeError, match="not installed"),
        ):
            download_kaggle_dataset("owner/dataset")

    def test_raises_when_credentials_missing(self, tmp_path: Path) -> None:
        kagglehub = ModuleType("kagglehub")
        kagglehub.dataset_download = lambda _: tmp_path / "unused"  # type: ignore[assignment]
        with (
            patch.dict(sys.modules, {"kagglehub": kagglehub}),
            patch("getitune.benchmark.dataset_helpers._has_kaggle_credentials", return_value=False),
            pytest.raises(RuntimeError, match="credentials not found"),
        ):
            download_kaggle_dataset("owner/dataset")

    def test_downloads_dataset_via_kagglehub(self, tmp_path: Path) -> None:
        kagglehub = ModuleType("kagglehub")
        kagglehub_path = tmp_path / "kagglehub_dataset"
        kagglehub_path.mkdir()
        kagglehub.dataset_download = lambda _: str(kagglehub_path)  # type: ignore[assignment]

        with (
            patch.dict(sys.modules, {"kagglehub": kagglehub}),
            patch("getitune.benchmark.dataset_helpers._has_kaggle_credentials", return_value=True),
        ):
            result = download_kaggle_dataset("owner/dataset")

        assert result == kagglehub_path
