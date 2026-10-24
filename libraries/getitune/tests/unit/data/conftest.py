# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from datumaro.experimental import Dataset

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.fixture
def fxt_mock_classification_dm_subset(mocker: MockerFixture) -> MagicMock:
    mock_dm_subset = mocker.MagicMock(spec=Dataset)
    mock_dm_subset.__len__.return_value = 1
    return mock_dm_subset


@pytest.fixture
def fxt_mock_detection_dm_subset(mocker: MockerFixture) -> MagicMock:
    mock_dm_subset = mocker.MagicMock(spec=Dataset)
    mock_dm_subset.__len__.return_value = 1
    return mock_dm_subset


@pytest.fixture
def fxt_mock_segmentation_dm_subset(mocker: MockerFixture) -> MagicMock:
    mock_dm_subset = mocker.MagicMock(spec=Dataset)
    mock_dm_subset.__len__.return_value = 1
    return mock_dm_subset
