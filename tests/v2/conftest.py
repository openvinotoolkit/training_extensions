"""OTX V2 Test codes."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Generator

import pytest


@pytest.fixture(scope="session")
def tmp_dir_path() -> Generator[Path, None, None]:
    with TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)
