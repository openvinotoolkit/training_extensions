"""Adapters for mmdeploy."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

AVAILABLE = True
VERSION = None
DEBUG = None


def raise_import_error(msg: str) -> ImportError:
    raise ImportError(msg)


try:
    import platform

    if platform.system() not in ("Linux", "Windows"):
        msg = "mmdeploy is only supports Windows and Linux."
        raise_import_error(msg)

    import mmdeploy

    VERSION = mmdeploy.__version__
except ImportError as e:
    AVAILABLE = False
    DEBUG = e
