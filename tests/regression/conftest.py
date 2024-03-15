# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import pytest

from tests.regression.summarize_test_results import summarize_results_data


@pytest.fixture(autouse=True, scope="session")
def run_regression_tests(tmp_dir_path):
    result_path = os.path.join(os.environ.get("REG_RESULTS_ROOT", tmp_dir_path), "reg_test_results")
    print(f"reg results path = {result_path}")
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    yield

    output_path = os.environ.get("TOX_WORK_DIR", os.getcwd())

    summarize_results_data(result_path, output_path)
