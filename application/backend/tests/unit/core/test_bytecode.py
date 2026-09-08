# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib
import importlib.util
import py_compile
import sys
from pathlib import Path

import pytest

from app.core import bytecode


@pytest.fixture
def guard():
    """Install the guard and restore the original loader afterwards."""
    original = bytecode._ORIGINAL_GET_CODE
    installed_before = bytecode._installed
    bytecode._installed = False
    bytecode.install_corrupt_bytecode_guard()
    yield
    from importlib.machinery import SourceFileLoader

    SourceFileLoader.get_code = original
    bytecode._installed = installed_before


def _write_module(tmp_path: Path, name: str) -> tuple[Path, Path]:
    """Create a module with a valid cache file, then corrupt the cache."""
    source = tmp_path / f"{name}.py"
    source.write_text("VALUE = 42\n")
    cache_path = py_compile.compile(str(source), doraise=True)
    assert cache_path is not None  # only None when compilation is skipped, never with doraise=True
    return source, Path(cache_path)


def _import_from(tmp_path: Path, name: str):
    sys.path.insert(0, str(tmp_path))
    try:
        sys.modules.pop(name, None)
        importlib.invalidate_caches()
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop(name, None)


def test_corrupted_cache_breaks_import_without_guard(tmp_path):
    source, cache = _write_module(tmp_path, "corrupt_without_guard")
    data = bytearray(cache.read_bytes())
    data[16:] = bytes(len(data) - 16)  # zero the marshalled payload, keep the valid header
    cache.write_bytes(bytes(data))

    with pytest.raises(ValueError, match="bad marshal data"):
        _import_from(tmp_path, "corrupt_without_guard")

    assert source.exists()


@pytest.mark.usefixtures("guard")
def test_guard_recompiles_module_with_corrupted_cache(tmp_path):
    _, cache = _write_module(tmp_path, "corrupt_with_guard")
    data = bytearray(cache.read_bytes())
    data[16:] = bytes(len(data) - 16)
    cache.write_bytes(bytes(data))

    module = _import_from(tmp_path, "corrupt_with_guard")

    assert module.VALUE == 42
    # The damaged cache file must be gone so the next import can write a healthy one.
    assert not cache.exists()


@pytest.mark.usefixtures("guard")
def test_guard_keeps_intact_modules_untouched(tmp_path):
    _, cache = _write_module(tmp_path, "healthy_module")
    original_bytes = cache.read_bytes()

    module = _import_from(tmp_path, "healthy_module")

    assert module.VALUE == 42
    assert cache.read_bytes() == original_bytes


def test_guard_reports_original_error_when_source_is_unavailable(monkeypatch, tmp_path):
    """If the source cannot be read either, the original marshal error must surface."""
    from importlib.machinery import SourceFileLoader

    def _raise_marshal_error(self, fullname):
        raise ValueError("bad marshal data (invalid reference)")

    monkeypatch.setattr(bytecode, "_ORIGINAL_GET_CODE", _raise_marshal_error)

    missing = tmp_path / "never_written.py"
    loader = SourceFileLoader("never_written", str(missing))

    with pytest.raises(ValueError, match="bad marshal data"):
        bytecode._get_code_with_recovery(loader, "never_written")


def test_guard_recovers_at_loader_level(monkeypatch, tmp_path):
    """The shim recompiles from source and drops the damaged cache file."""
    from importlib.machinery import SourceFileLoader

    def _raise_marshal_error(self, fullname):
        raise ValueError("bad marshal data (invalid reference)")

    monkeypatch.setattr(bytecode, "_ORIGINAL_GET_CODE", _raise_marshal_error)

    source, cache = _write_module(tmp_path, "loader_level")
    assert cache.exists()

    code = bytecode._get_code_with_recovery(SourceFileLoader("loader_level", str(source)), "loader_level")
    assert code is not None

    namespace: dict = {}
    exec(code, namespace)
    assert namespace["VALUE"] == 42
    assert not cache.exists()


def test_install_is_idempotent():
    original = bytecode._ORIGINAL_GET_CODE
    installed_before = bytecode._installed
    bytecode._installed = False
    try:
        assert bytecode.install_corrupt_bytecode_guard() is True
        assert bytecode.install_corrupt_bytecode_guard() is False
    finally:
        from importlib.machinery import SourceFileLoader

        SourceFileLoader.get_code = original
        bytecode._installed = installed_before
