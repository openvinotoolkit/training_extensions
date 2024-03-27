"""OTX Classification perfomance tests."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest

from otx.cli.registry import Registry
from typing import Callable
from .benchmark import Benchmark


MODEL_TEMPLATES = Registry(f"src/otx/algorithms").filter(task_type="CLASSIFICATION").templates
MODEL_IDS = [template.model_template_id for template in MODEL_TEMPLATES]


class TestPerfSingleLabelClassification:
    """Benchmark single-label classification."""

    BENCHMARK_CONFIGS = {
        "small": {
            "tags": {
                "task": "single_label_classification",
            },
            "datasets": [
                "classification/single_label/multiclass_CUB_small/1",
                "classification/single_label/multiclass_CUB_small/2",
                "classification/single_label/multiclass_CUB_small/3",
            ],
            "num_repeat": 5,
        },
        "medium": {
            "tags": {
                "task": "single_label_classification",
            },
            "datasets": [
                "classification/single_label/multiclass_CUB_medium",
            ],
            "num_repeat": 5,
        },
        "large": {
            "tags": {
                "task": "single_label_classification",
            },
            "datasets": [
                "classification/single_label/multiclass_food101_large",
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


class TestPerfMultiLabelClassification:
    """Benchmark multi-label classification."""

    BENCHMARK_CONFIGS = {
        "small": {
            "tags": {
                "task": "multi_label_classification",
            },
            "datasets": [
                "classification/multi_label/multilabel_CUB_small/1",
                "classification/multi_label/multilabel_CUB_small/2",
                "classification/multi_label/multilabel_CUB_small/3",
            ],
            "num_repeat": 5,
        },
        "medium": {
            "tags": {
                "task": "multi_label_classification",
            },
            "datasets": [
                "classification/multi_label/multilabel_CUB_medium",
            ],
            "num_repeat": 5,
        },
        "large": {
            "tags": {
                "task": "multi_label_classification",
            },
            "datasets": [
                "classification/multi_label/multilabel_food101_large",
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


class TestPerfHierarchicalLabelClassification:
    """Benchmark hierarchcial-label classification."""

    BENCHMARK_CONFIGS = {
        "small": {
            "tags": {
                "task": "hierarchical_label_classification",
            },
            "datasets": [
                "classification/h_label/h_label_CUB_small/1",
                "classification/h_label/h_label_CUB_small/2",
                "classification/h_label/h_label_CUB_small/3",
            ],
            "num_repeat": 5,
        },
        "medium": {
            "tags": {
                "task": "hierarchical_label_classification",
            },
            "datasets": [
                "classification/h_label/h_label_CUB_medium",
            ],
            "num_repeat": 5,
        },
        # TODO: Add large dataset
        # "large": {
        #     "tags": {
        #         "task": "hierarchical_label_classification",
        #     },
        #     "datasets": [
        #     ],
        #     "num_repeat": 5,
        # },
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
