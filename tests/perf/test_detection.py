"""OTX Detection perfomance tests."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest

from otx.cli.registry import Registry
from typing import Callable
from .benchmark import Benchmark


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
            "num_repeat": 5,
        },
        "medium": {
            "tags": {
                "task": "detection",
            },
            "datasets": [
                "detection/pothole_medium",
            ],
            "num_repeat": 5,
        },
        "large": {
            "tags": {
                "task": "detection",
            },
            "datasets": [
                "detection/vitens_large",
            ],
            "num_repeat": 5,
        },
    }

    @pytest.mark.parametrize("fxt_model_id", MODEL_TEMPLATES, ids=MODEL_IDS, indirect=True)
    @pytest.mark.parametrize("fxt_benchmark", BENCHMARK_CONFIGS.items(), ids=BENCHMARK_CONFIGS.keys(), indirect=True)
    def test_perf(self, fxt_model_id: str, fxt_benchmark: Benchmark):
        """Benchmark performance metrics."""
        result = fxt_benchmark.run(model_id=fxt_model_id)
        fxt_benchmark.check(
            result,
            criteria=[
                {
                    "name": "f-measure(train)",
                    "op": ">",
                    "margin": 0.1,
                },
                {
                    "name": "epoch",
                    "op": "<",
                    "margin": 0.1,
                },
                {
                    "name": "f-measure(export)",
                    "op": ">",
                    "margin": 0.1,
                },
                {
                    "name": "f-measure(optimize)",
                    "op": ">",
                    "margin": 0.1,
                },
                {
                    "name": "train_e2e_time",
                    "op": "<",
                    "margin": 0.1,
                },
                {
                    "name": "avg_data_time",
                    "op": "<",
                    "margin": 0.1,
                },
                {
                    "name": "avg_iter_time",
                    "op": "<",
                    "margin": 0.1,
                },
                {
                    "name": "avg_time_per_image(export)",
                    "op": "<",
                    "margin": 0.1,
                },
                {
                    "name": "avg_time_per_image(optimize)",
                    "op": "<",
                    "margin": 0.1,
                },
            ],
        )
