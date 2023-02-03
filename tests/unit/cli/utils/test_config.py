# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os
import tempfile

import pytest

from otx.cli.utils.config import configure_dataset, override_parameters
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
def test_override_parameters():
    parameters = {"default_value": 1, "value": 2}
    overrides = {"default_value": 3}

    override_parameters(overrides, parameters)
    assert parameters["default_value"] == 3
    assert parameters["value"] == 2

    overrides = {"unknown_key": 3}
    with pytest.raises(ValueError) as excinfo:
        override_parameters(overrides, parameters)
    assert 'The "unknown_key" is not in allowed_keys' in str(excinfo.value)

    overrides = {"value": {"new_key": 3}}
    with pytest.raises(ValueError) as excinfo:
        override_parameters(overrides, parameters)
    assert 'The "new_key" is not in allowed_keys' in str(excinfo.value)


@e2e_pytest_unit
def test_configure_dataset():
    data_file = tempfile.NamedTemporaryFile(delete=False)
    data_file.write(
        "data:\n  \
            train:\n    ann-files: []\n    data-roots: []\n  \
            val:\n    ann-files: []\n    data-roots: []\n  \
            test:\n    ann-files: []\n    data-roots: []\n  \
            unlabeled:\n    file-list: []\n    data-roots: []".encode()
    )
    data_file.close()

    class Args:
        data = data_file.name
        train_ann_files = [1, 2, 3]
        val_ann_files = [4, 5, 6]
        test_ann_files = [7, 8, 9]
        unlabeled_file_list = [10, 11, 12]

        def __iter__(self):
            for attr in dir(self):
                if not attr.startswith("__"):
                    yield attr

    data_config = configure_dataset(Args())
    assert data_config["data"]["train"]["ann-files"] == [1, 2, 3]
    assert data_config["data"]["val"]["ann-files"] == [4, 5, 6]
    assert data_config["data"]["test"]["ann-files"] == [7, 8, 9]
    assert data_config["data"]["unlabeled"]["file-list"] == [10, 11, 12]

    os.unlink(data_file.name)
