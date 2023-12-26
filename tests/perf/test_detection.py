"""OTX Detection perfomance tests."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest

from otx.cli.registry import Registry
from .benchmark import OTXBenchmark


MODEL_TEMPLATES = Registry(f"src/otx/algorithms").filter(task_type="DETECTION").templates
MODEL_IDS = [template.model_template_id for template in MODEL_TEMPLATES]


class TestPerfDetection:
    """Benchmark basic object detection."""

    BENCHMARK_CONFIGS = {
        "small": {
            "tags": {
                "task": "detection",
            },
            "datasets": [
                "detection/pothole_small/1",
                "detection/pothole_small/2",
                "detection/pothole_small/3",
            ],
            "num_repeat": 3,
        },
        "medium": {
            "tags": {
                "task": "detection",
            },
            "datasets": [
                "detection/pothole_medium",
            ],
            "num_repeat": 3,
        },
        "large": {
            "tags": {
                "task": "detection",
            },
            "datasets": [
                "detection/vitens_large",
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
