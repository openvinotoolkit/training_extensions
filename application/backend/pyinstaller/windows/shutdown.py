# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyInstaller runtime hook: expose a Win32 shutdown event to the desktop shell.

The Tauri shell must be able to ask the backend to stop *gracefully* before it
exits, so the ASGI lifespan shutdown can run (finish in-flight requests, stop
worker processes, checkpoint and close SQLite). A hard kill instead leaves
WAL/journal files behind and can truncate an in-flight migration or training
checkpoint, which surfaces as instability on the next launch.

Console control events are not usable for this. The shell is a GUI process with
no console of its own, and the packaged side-car is started with
``CREATE_NO_WINDOW``, which leaves the child without a console handle - so there
is no console for ``GenerateConsoleCtrlEvent`` to travel through.

This hook therefore publishes a named Win32 event, ``Local\\<prefix><pid>``, and
waits on it in a daemon thread. The ``Local\\`` namespace is per-session, so the
event is reachable by the shell (same user session) and by nothing outside it;
the MSIX package is full-trust, hence not AppContainer-isolated, so named-object
access behaves normally.

Setting the event is translated into a plain :class:`threading.Event`, published
as ``geti_shutdown.shutdown_requested``. ``app.main`` waits on *that* and stays
free of any Windows-specific machinery: all the platform plumbing lives here.

Keep :data:`SHUTDOWN_EVENT_PREFIX` in sync with ``SHUTDOWN_EVENT_PREFIX`` in
``application/ui/src-tauri/src/job.rs``, which owns the other half of this
contract.
"""

import ctypes
import os
import sys
import threading
import types
from ctypes import wintypes

#: Prefix of the per-process event name. The full name is ``Local\<prefix><pid>``.
SHUTDOWN_EVENT_PREFIX = "geti-backend-shutdown-"

#: Synthetic module through which the request is handed to the application.
BRIDGE_MODULE_NAME = "geti_shutdown"

#: Attribute on the bridge module holding the :class:`threading.Event`.
BRIDGE_ATTRIBUTE = "shutdown_requested"

#: ``WaitForSingleObject`` timeout meaning "wait forever".
_INFINITE = 0xFFFFFFFF


def _publish_bridge() -> threading.Event:
    """Publish the shutdown request as an importable module attribute.

    A synthetic module is used rather than an environment variable or a file so
    the application can pick the request up with no polling and no knowledge of
    how it was produced.

    Returns:
        The event the application will wait on.
    """
    request = threading.Event()
    module = types.ModuleType(BRIDGE_MODULE_NAME)
    setattr(module, BRIDGE_ATTRIBUTE, request)
    sys.modules[BRIDGE_MODULE_NAME] = module
    return request


def _create_shutdown_event(name: str) -> int | None:
    """Create the manual-reset Win32 event the shell signals.

    Args:
        name: Fully qualified object name, including the ``Local\\`` namespace.

    Returns:
        The event handle, or ``None`` when it could not be created.
    """
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    kernel32.CreateEventW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
    kernel32.CreateEventW.restype = wintypes.HANDLE

    # Resolved up front: looking a symbol up goes through ctypes, which may
    # itself overwrite the thread's last-error value we want to report. Reading
    # it via kernel32 rather than ``ctypes.get_last_error`` also keeps this file
    # free of ctypes attributes that only exist on Windows.
    get_last_error = kernel32.GetLastError
    get_last_error.argtypes = []
    get_last_error.restype = wintypes.DWORD

    # Manual reset, initially unsignalled: the shell only ever sets it once.
    handle = kernel32.CreateEventW(None, True, False, name)
    if not handle:
        print(f"Setup Hook: Could not create shutdown event {name}: error {get_last_error()}")
        return None
    return handle


def _wait_in_background(handle: int, request: threading.Event) -> None:
    """Set ``request`` once ``handle`` is signalled, from a daemon thread.

    The thread is a daemon so it can never delay interpreter exit, and it blocks
    on the kernel object rather than polling.
    """
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD

    def _wait() -> None:
        kernel32.WaitForSingleObject(handle, _INFINITE)
        request.set()

    threading.Thread(target=_wait, name="shutdown-event-listener", daemon=True).start()


def _is_multiprocessing_child() -> bool:
    """Whether this interpreter is a multiprocessing worker rather than the server.

    Runtime hooks also run in spawned children. Only the process that serves HTTP
    is ever addressed by the shell, so workers skip the event entirely instead of
    each carrying a redundant handle and thread.
    """
    return "--multiprocessing-fork" in sys.argv


def _main() -> None:
    if _is_multiprocessing_child():
        return

    # Published unconditionally: an event that is never set is harmless and keeps
    # the application side free of "is this available?" branching.
    request = _publish_bridge()

    name = f"Local\\{SHUTDOWN_EVENT_PREFIX}{os.getpid()}"
    handle = _create_shutdown_event(name)
    if handle is None:
        return

    _wait_in_background(handle, request)
    print(f"Setup Hook: Listening for shutdown requests on {name}")


_main()
