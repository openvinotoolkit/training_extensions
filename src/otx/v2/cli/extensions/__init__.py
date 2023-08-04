"""OTX CLI Extensions."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from .doctor import add_doctor_parser
from .doctor import doctor as doctor_main
from .install import add_install_parser
from .install import install as install_main

CLI_EXTENSIONS: dict[str, dict] = {
    "doctor": {
        "add_parser": add_doctor_parser,
        "main": doctor_main,
    },
    "install": {
        "add_parser": add_install_parser,
        "main": install_main,
    },
}
