# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyInstaller runtime hook: suppress Windows' modal DLL loader error dialogs.

OpenVINO's GPU plugin links against OpenCL 3.0 entry points. On machines with an
older OpenCL ICD loader, loading the plugin fails and Windows shows a blocking
"Entry Point Not Found" dialog on every launch. With ``SEM_FAILCRITICALERRORS``
the load fails silently instead, OpenVINO skips the GPU device and the backend
keeps running. Child processes inherit the error mode.
"""

import ctypes

SEM_FAILCRITICALERRORS = 0x0001
SEM_NOOPENFILEERRORBOX = 0x8000

_kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
_kernel32.SetErrorMode(_kernel32.GetErrorMode() | SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX)
