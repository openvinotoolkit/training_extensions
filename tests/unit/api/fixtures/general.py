"""
General fixtures.
"""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from tests.test_helpers import LabelSchemaExample


@pytest.fixture(scope="session")
def label_schema_example():
    """
    Returns a label schema example.
    """

    return LabelSchemaExample()
