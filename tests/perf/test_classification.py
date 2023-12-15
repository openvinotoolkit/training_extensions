"""OTX Classification Perfomance tests."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest

from otx.cli.registry import Registry
from tests.test_suite.run_test_command import check_run


TEMPLATES = Registry(f"src/otx/algorithms").filter(task_type="CLASSIFICATION").templates
TEMPLATE_NAMES = [template.name for template in TEMPLATES]


class TestPerfSingleLabelClassification:
    BENCHMARK_CONFIGS = {
        "small": {
            "datasets": [
                "classification/single_label/multiclass_CUB_small/1",
                "classification/single_label/multiclass_CUB_small/2",
                "classification/single_label/multiclass_CUB_small/3",
            ],
            "num_repeat": 3,
        },
        "medium": {
            "datasets": [
                "classification/single_label/multiclass_CUB_medium",
            ],
            "num_repeat": 3,
        },
        "large": {
            "datasets": [
                "classification/single_label/multiclass_food101_large",
            ],
            "num_repeat": 1,
        },
    }

    @pytest.mark.parametrize("fxt_template", TEMPLATES, ids=TEMPLATE_NAMES, indirect=True)
    @pytest.mark.parametrize("fxt_benchmark_config", BENCHMARK_CONFIGS.items(), ids=BENCHMARK_CONFIGS.keys(), indirect=True)
    def test_accuarcy(self, fxt_template, fxt_benchmark_config, fxt_build_command):
        """Benchmark accruacy metrics."""
        model_template = fxt_template
        data_size, datasets, num_epoch, num_repeat = fxt_benchmark_config
        tag = f"singlelabel-classification-accuracy-{data_size}"
        command = fxt_build_command(
            tag,
            model_template,
            datasets,
            num_epoch,
            num_repeat,
        )
        check_run(command)

    @pytest.mark.parametrize("fxt_template", TEMPLATES, ids=TEMPLATE_NAMES, indirect=True)
    @pytest.mark.parametrize("fxt_benchmark_config", BENCHMARK_CONFIGS.items(), ids=BENCHMARK_CONFIGS.keys(), indirect=True)
    def test_speed(self, fxt_template, fxt_benchmark_config, fxt_build_command):
        """Benchmark train time per iter / infer time per image."""
        model_template = fxt_template
        data_size, datasets, num_epoch, num_repeat = fxt_benchmark_config
        # Override default iteration setting, in case there's no user input
        # "--data-size large -k speed" is recommended.
        if num_epoch == 0:
            num_epoch = 2
        if num_repeat == 0:
            num_repeat = 1
        tag = f"singlelabel-classification-speed-{data_size}"
        command = fxt_build_command(
            tag,
            model_template,
            datasets,
            num_epoch,
            num_repeat,
            track_resources=True,  # Measure CPU/GPU usages
        )
        check_run(command)
