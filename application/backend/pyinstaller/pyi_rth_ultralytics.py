# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyInstaller runtime hook: stop Ultralytics from pip-installing packages at runtime.

Ultralytics' ``check_requirements`` runs ``pip install`` for missing optional
dependencies. In the frozen app that would install into whatever Python happens
to be on ``PATH`` (or fail after a network round-trip); dependencies must come
from the bundle instead.
"""

import os

os.environ["YOLO_AUTOINSTALL"] = "false"
