"""OTX Action perfomance tests."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest

from otx.cli.registry import Registry
from typing import Callable
from .benchmark import Benchmark


class TestPerfActionClassification:
    """Benchmark action classification."""

    MODEL_TEMPLATES = Registry(f"src/otx/algorithms").filter(task_type="ACTION_CLASSIFICATION").templates
    MODEL_IDS = [template.model_template_id for template in MODEL_TEMPLATES]

    BENCHMARK_CONFIGS = {
        "small": {
            "tags": {
                "task": "action_classification",
            },
            "datasets": [
                "action/action_classification/ucf_cvat_5percent",
            ],
            "num_repeat": 5,
            "num_epoch": 10,
        },
        "medium": {
            "tags": {
                "task": "action_classification",
            },
            "datasets": [
                "action/action_classification/ucf_cvat_30percent",
            ],
            "num_repeat": 5,
            "num_epoch": 10,
        },
        "large": {
            "tags": {
                "task": "action_classification",
            },
            "datasets": [
                "action/action_classification/ucf_cvat",
            ],
            "num_repeat": 5,
            "num_epoch": 3,
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
                    "name": "Accuracy(train)",
                    "op": ">",
                    "margin": 0.1,
                },
                {
                    "name": "Accuracy(export)",
                    "op": ">",
                    "margin": 0.1,
                },
                {
                    "name": "Accuracy(optimize)",
                    "op": ">",
                    "margin": 0.1,
                },
                {
                    "name": "epoch",
                    "op": "<",
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


class TestPerfActionDetection:
    """Benchmark action detection."""

    MODEL_TEMPLATES = Registry(f"src/otx/algorithms").filter(task_type="ACTION_DETECTION").templates
    MODEL_IDS = [template.model_template_id for template in MODEL_TEMPLATES]

    BENCHMARK_CONFIGS = {
        "small": {
            "tags": {
                "task": "action_detection",
            },
            "datasets": [
                "action/action_detection/UCF101_cvat_5percent",
            ],
            "num_repeat": 5,
            "num_epoch": 3,
        },
        "medium": {
            "tags": {
                "task": "action_detection",
            },
            "datasets": [
                "action/action_detection/UCF101_cvat_30percent",
            ],
            "num_repeat": 5,
            "num_epoch": 3,
        },
        "large": {
            "tags": {
                "task": "action_detection",
            },
            "datasets": [
                "action/action_detection/UCF101_cvat",
            ],
            "num_repeat": 5,
            "num_epoch": 1,
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
