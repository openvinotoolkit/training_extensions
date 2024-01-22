"""OTX Anomaly perfomance tests."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest

from otx.cli.registry import Registry
from typing import Callable
from .benchmark import OTXBenchmark


class TestPerfAnomalyClassification:
    """Benchmark anomaly classification."""

    MODEL_TEMPLATES = Registry(f"src/otx/algorithms").filter(task_type="ANOMALY_CLASSIFICATION").templates
    MODEL_IDS = [template.model_template_id for template in MODEL_TEMPLATES]

    BENCHMARK_CONFIGS = {
        "small": {
            "tags": {
                "task": "anomaly_classification",
            },
            "datasets": [
                "anomaly/mvtec/bottle_small/1",
                "anomaly/mvtec/bottle_small/2",
                "anomaly/mvtec/bottle_small/3",
            ],
            "num_repeat": 3,
        },
        "medium": {
            "tags": {
                "task": "anomaly_classification",
            },
            "datasets": [
                "anomaly/mvtec/wood_medium",
            ],
            "num_repeat": 3,
        },
        "large": {
            "tags": {
                "task": "anomaly_classification",
            },
            "datasets": [
                "anomaly/mvtec/hazelnut_large",
            ],
            "num_repeat": 1,
        },
    }

    @pytest.mark.parametrize("fxt_model_id", MODEL_TEMPLATES, ids=MODEL_IDS, indirect=True)
    @pytest.mark.parametrize("fxt_benchmark", BENCHMARK_CONFIGS.items(), ids=BENCHMARK_CONFIGS.keys(), indirect=True)
    def test_accuracy(self, fxt_model_id: str, fxt_benchmark: OTXBenchmark, fxt_check_benchmark_result: Callable):
        """Benchmark accruacy metrics."""
        result = fxt_benchmark.run(
            model_id=fxt_model_id,
            tags={"benchmark": "accuracy"},
        )
        fxt_check_benchmark_result(
            result,
            key=("accuracy", fxt_benchmark.tags["task"], fxt_benchmark.tags["data_size"], fxt_model_id),
            checks=[
                {
                    "name": "f-measure(train)",
                    "op": ">",
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
            ],
        )

    @pytest.mark.parametrize("fxt_model_id", MODEL_TEMPLATES, ids=MODEL_IDS, indirect=True)
    @pytest.mark.parametrize("fxt_benchmark", BENCHMARK_CONFIGS.items(), ids=BENCHMARK_CONFIGS.keys(), indirect=True)
    def test_speed(self, fxt_model_id: str, fxt_benchmark: OTXBenchmark, fxt_check_benchmark_result: Callable):
        """Benchmark train time per iter / infer time per image."""
        fxt_benchmark.track_resources = True
        result = fxt_benchmark.run(
            model_id=fxt_model_id,
            tags={"benchmark": "speed"},
        )
        fxt_check_benchmark_result(
            result,
            key=("speed", fxt_benchmark.tags["task"], fxt_benchmark.tags["data_size"], fxt_model_id),
            checks=[
                {
                    "name": "train_e2e_time",
                    "op": "<",
                    "margin": 0.1,
                },
            ],
        )


class TestPerfAnomalyDetection:
    """Benchmark anomaly detection."""

    MODEL_TEMPLATES = Registry(f"src/otx/algorithms").filter(task_type="ANOMALY_DETECTION").templates
    MODEL_IDS = [template.model_template_id for template in MODEL_TEMPLATES]

    BENCHMARK_CONFIGS = {
        "small": {
            "tags": {
                "task": "anomaly_detection",
            },
            "datasets": [
                "anomaly/mvtec/bottle_small/1",
                "anomaly/mvtec/bottle_small/2",
                "anomaly/mvtec/bottle_small/3",
            ],
            "num_repeat": 3,
        },
        "medium": {
            "tags": {
                "task": "anomaly_detection",
            },
            "datasets": [
                "anomaly/mvtec/wood_medium",
            ],
            "num_repeat": 3,
        },
        "large": {
            "tags": {
                "task": "anomaly_detection",
            },
            "datasets": [
                "anomaly/mvtec/hazelnut_large",
            ],
            "num_repeat": 1,
        },
    }

    @pytest.mark.parametrize("fxt_model_id", MODEL_TEMPLATES, ids=MODEL_IDS, indirect=True)
    @pytest.mark.parametrize("fxt_benchmark", BENCHMARK_CONFIGS.items(), ids=BENCHMARK_CONFIGS.keys(), indirect=True)
    def test_accuracy(self, fxt_model_id: str, fxt_benchmark: OTXBenchmark, fxt_check_benchmark_result: Callable):
        """Benchmark accruacy metrics."""
        result = fxt_benchmark.run(
            model_id=fxt_model_id,
            tags={"benchmark": "accuracy"},
        )
        fxt_check_benchmark_result(
            result,
            key=("accuracy", fxt_benchmark.tags["task"], fxt_benchmark.tags["data_size"], fxt_model_id),
            checks=[
                {
                    "name": "f-measure(train)",
                    "op": ">",
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
            ],
        )

    @pytest.mark.parametrize("fxt_model_id", MODEL_TEMPLATES, ids=MODEL_IDS, indirect=True)
    @pytest.mark.parametrize("fxt_benchmark", BENCHMARK_CONFIGS.items(), ids=BENCHMARK_CONFIGS.keys(), indirect=True)
    def test_speed(self, fxt_model_id: str, fxt_benchmark: OTXBenchmark, fxt_check_benchmark_result: Callable):
        """Benchmark train time per iter / infer time per image."""
        fxt_benchmark.track_resources = True
        result = fxt_benchmark.run(
            model_id=fxt_model_id,
            tags={"benchmark": "speed"},
        )
        fxt_check_benchmark_result(
            result,
            key=("speed", fxt_benchmark.tags["task"], fxt_benchmark.tags["data_size"], fxt_model_id),
            checks=[
                {
                    "name": "train_e2e_time",
                    "op": "<",
                    "margin": 0.1,
                },
            ],
        )


class TestPerfAnomalySegmentation:
    """Benchmark anomaly segmentation."""

    MODEL_TEMPLATES = Registry(f"src/otx/algorithms").filter(task_type="ANOMALY_SEGMENTATION").templates
    MODEL_IDS = [template.model_template_id for template in MODEL_TEMPLATES]

    BENCHMARK_CONFIGS = {
        "small": {
            "tags": {
                "task": "anomaly_segmentation",
            },
            "datasets": [
                "anomaly/mvtec/bottle_small/1",
                "anomaly/mvtec/bottle_small/2",
                "anomaly/mvtec/bottle_small/3",
            ],
            "num_repeat": 3,
        },
        "medium": {
            "tags": {
                "task": "anomaly_segmentation",
            },
            "datasets": [
                "anomaly/mvtec/wood_medium",
            ],
            "num_repeat": 3,
        },
        "large": {
            "tags": {
                "task": "anomaly_segmentation",
            },
            "datasets": [
                "anomaly/mvtec/hazelnut_large",
            ],
            "num_repeat": 1,
        },
    }

    @pytest.mark.parametrize("fxt_model_id", MODEL_TEMPLATES, ids=MODEL_IDS, indirect=True)
    @pytest.mark.parametrize("fxt_benchmark", BENCHMARK_CONFIGS.items(), ids=BENCHMARK_CONFIGS.keys(), indirect=True)
    def test_accuracy(self, fxt_model_id: str, fxt_benchmark: OTXBenchmark, fxt_check_benchmark_result: Callable):
        """Benchmark accruacy metrics."""
        result = fxt_benchmark.run(
            model_id=fxt_model_id,
            tags={"benchmark": "accuracy"},
        )
        fxt_check_benchmark_result(
            result,
            key=("accuracy", fxt_benchmark.tags["task"], fxt_benchmark.tags["data_size"], fxt_model_id),
            checks=[
                {
                    "name": "f-measure(train)",
                    "op": ">",
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
            ],
        )

    @pytest.mark.parametrize("fxt_model_id", MODEL_TEMPLATES, ids=MODEL_IDS, indirect=True)
    @pytest.mark.parametrize("fxt_benchmark", BENCHMARK_CONFIGS.items(), ids=BENCHMARK_CONFIGS.keys(), indirect=True)
    def test_speed(self, fxt_model_id: str, fxt_benchmark: OTXBenchmark, fxt_check_benchmark_result: Callable):
        """Benchmark train time per iter / infer time per image."""
        fxt_benchmark.track_resources = True
        result = fxt_benchmark.run(
            model_id=fxt_model_id,
            tags={"benchmark": "speed"},
        )
        fxt_check_benchmark_result(
            result,
            key=("speed", fxt_benchmark.tags["task"], fxt_benchmark.tags["data_size"], fxt_model_id),
            checks=[
                {
                    "name": "train_e2e_time",
                    "op": "<",
                    "margin": 0.1,
                },
            ],
        )
