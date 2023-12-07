# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from otx.core.config import register_configs


@pytest.fixture(scope="session", autouse=True)
def fxt_register_configs() -> None:
    register_configs()
