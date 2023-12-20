"""OTX Classification Perfomance tests."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest

from otx.cli.registry import Registry
from .benchmark import OTXBenchmark


MODEL_TEMPLATES = Registry(f"src/otx/algorithms").filter(task_type="CLASSIFICATION").templates
MODEL_IDS = [template.model_template_id for template in MODEL_TEMPLATES]


class TestPerfSingleLabelClassification:
    BENCHMARK_CONFIGS = {
        "small": {
            "tags": {
                "task": "single-label-classification",
            },
            "datasets": [
                "classification/single_label/multiclass_CUB_small/1",
                "classification/single_label/multiclass_CUB_small/2",
                "classification/single_label/multiclass_CUB_small/3",
            ],
            "num_repeat": 3,
        },
        "medium": {
            "tags": {
                "task": "single-label-classification",
            },
            "datasets": [
                "classification/single_label/multiclass_CUB_medium",
            ],
            "num_repeat": 3,
        },
        "large": {
            "tags": {
                "task": "single-label-classification",
            },
            "datasets": [
                "classification/single_label/multiclass_food101_large",
            ],
            "num_repeat": 1,
        },
    }

    @pytest.mark.parametrize("fxt_model_id", MODEL_TEMPLATES, ids=MODEL_IDS, indirect=True)
    @pytest.mark.parametrize("fxt_benchmark", BENCHMARK_CONFIGS.items(), ids=BENCHMARK_CONFIGS.keys(), indirect=True)
    def test_accuracy(self, fxt_model_id: str, fxt_benchmark: OTXBenchmark):
        """Benchmark accruacy metrics."""
        result = fxt_benchmark.run(
            model_id=fxt_model_id,
            tags={"benchmark": "accuracy"},
        )

    @pytest.mark.parametrize("fxt_model_id", MODEL_TEMPLATES, ids=MODEL_IDS, indirect=True)
    @pytest.mark.parametrize("fxt_benchmark", BENCHMARK_CONFIGS.items(), ids=BENCHMARK_CONFIGS.keys(), indirect=True)
    def test_speed(self, fxt_model_id: str, fxt_benchmark: OTXBenchmark):
        """Benchmark train time per iter / infer time per image."""
        fxt_benchmark.track_resources = True
        result = fxt_benchmark.run(
            model_id=fxt_model_id,
            tags={"benchmark": "speed"},
        )
