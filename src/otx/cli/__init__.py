# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoints."""
from datetime import timedelta
from time import time

from otx.cli.cli import OTXCLI


def main() -> None:
    """Entry point for OTX CLI.

    This function is a single entry point for all OTX CLI related operations:
    """
    start = time()
    OTXCLI()
    dt = timedelta(seconds=time() - start)
    print(f"Elapsed time: {dt}")


if __name__ == "__main__":
    main()
