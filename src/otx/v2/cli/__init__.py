"""OTX CLI module."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from otx.v2 import OTX_LOGO


def print_otx_logo() -> None:
    print()
    print("\033[0;35m" + OTX_LOGO + "\033[0m")
    print()
