"""OTX Classification Perfomance tests."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest

from otx.cli.registry import Registry
from tests.test_suite.run_test_command import check_run


templates = Registry(f"src/otx/algorithms").filter(task_type="CLASSIFICATION").templates
templates_names = [template.name for template in templates]


class TestPerfMultiClassClassification:
    data_settings = {
        "small": {
            "datasets": [
                "small_dataset/1",
                "small_dataset/2",
                "small_dataset/3",
            ],
            "num_repeat": 3,
        },
        "medium": {
            "datasets": [
                "medium_dataset",
            ],
            "num_repeat": 3,
        },
        "large": {
            "datasets": [
                "large_dataset",
            ],
            "num_repeat": 1,
        },
    }
    @pytest.mark.parametrize("fxt_template", templates, ids=templates_names, indirect=True)
    @pytest.mark.parametrize("fxt_data_setting", data_settings.items(), ids=data_settings.keys(), indirect=True)
    def test_benchmark(self, fxt_template, fxt_data_setting, fxt_build_command):
        model_template = fxt_template
        data_size, datasets, num_repeat = fxt_data_setting
        tag = f"multiclass-classification-{data_size}"
        command = fxt_build_command(tag, model_template, datasets, num_repeat)
        check_run(command)
