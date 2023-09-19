"""OTX CLI Extensions."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from typing import Dict

from .doctor import add_doctor_parser, doctor
from .install import add_install_parser, install

CLI_EXTENSIONS: Dict[str, Dict] = {
    "doctor": {
        "add_parser": add_doctor_parser,
        "main": doctor,
    },
    "install": {
        "add_parser": add_install_parser,
        "main": install,
    },
}
