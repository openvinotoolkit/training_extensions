"""CLI main function."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .cli import OTXCLI


def main() -> None:
    """Entry point for OTX CLI.

    This function is a single entry point for all OTX CLI related operations:
    """
    OTXCLI()


if __name__ == "__main__":
    main()
