# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Recovery from corrupted cached bytecode (``.pyc``) files.

The shipped virtual environment is pre-compiled at image build time
(``UV_COMPILE_BYTECODE=1``), so every module in ``site-packages`` has a ``__pycache__``
entry that CPython loads instead of the source. When one of those cache files is damaged
afterwards - a truncated or partially zeroed file caused by an unclean shutdown, a full
disk, or a corrupted container/image layer - the import machinery aborts with an opaque::

    ValueError: bad marshal data (invalid reference)

raised from ``importlib._bootstrap_external._compile_bytecode``. The failure surfaces
wherever the module happens to be imported first, which for the heavy training stack is in
the middle of a job (e.g. ``torch.onnx.ops._impl`` while preparing a training
configuration), and it keeps recurring because nothing ever invalidates the broken cache.

The matching ``.py`` source is untouched in these cases, so the situation is fully
recoverable: discard the damaged cache file and compile the source again. This module
installs a small shim around :meth:`importlib.machinery.SourceFileLoader.get_code` that
does exactly that and logs a warning, so the underlying corruption stays visible instead of
turning into a hard failure of the job.
"""

from __future__ import annotations

import os
from importlib.machinery import SourceFileLoader
from importlib.util import cache_from_source
from types import CodeType

from loguru import logger

# Bound before any patching, so re-installation can never chain the shim onto itself.
_ORIGINAL_GET_CODE = SourceFileLoader.get_code

_installed = False


def _cache_path_for(source_path: str) -> str | None:
    """Return the ``__pycache__`` path for a source file, or None if it has none."""
    try:
        return cache_from_source(source_path)
    except (NotImplementedError, ValueError):
        return None


def _discard_cache_file(cache_path: str) -> None:
    """Delete a damaged cache file, tolerating a read-only or already-cleaned location."""
    try:
        os.unlink(cache_path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        # Recompiling from source still works, the next import will just pay the cost again.
        logger.warning("Could not remove corrupted bytecode cache '{}': {}", cache_path, exc)


def _get_code_with_recovery(self: SourceFileLoader, fullname: str) -> CodeType | None:
    """``SourceFileLoader.get_code`` that falls back to the source when the cache is corrupt."""
    try:
        return _ORIGINAL_GET_CODE(self, fullname)
    except (ValueError, EOFError) as exc:
        # ValueError/EOFError here means marshal choked on the cached bytecode; a stale or
        # wrong-magic cache raises ImportError instead and is already handled by CPython.
        try:
            source_path = self.get_filename(fullname)
            cache_path = _cache_path_for(source_path)
            source_bytes = self.get_data(source_path)
        except Exception:
            raise exc from None

        logger.warning(
            "Corrupted bytecode cache detected for module '{}' ({}): {}. Removing it and recompiling from source '{}'.",
            fullname,
            cache_path or "<no cache path>",
            exc,
            source_path,
        )
        if cache_path is not None:
            _discard_cache_file(cache_path)

        try:
            return self.source_to_code(source_bytes, source_path)  # type: ignore[attr-defined,no-any-return]
        except Exception:
            raise exc from None


def install_corrupt_bytecode_guard() -> bool:
    """Make imports resilient against damaged ``.pyc`` files.

    Patches the standard source-file loader so that a module whose cached bytecode cannot be
    unmarshalled is transparently recompiled from its (intact) source instead of aborting the
    import with ``ValueError: bad marshal data``.

    Returns:
        bool: True if the guard was installed by this call, False if it was already active.
    """
    global _installed  # noqa: PLW0603 - process-wide, one-shot patch of the import machinery
    if _installed:
        return False

    SourceFileLoader.get_code = _get_code_with_recovery  # type: ignore[method-assign]
    _installed = True
    return True
