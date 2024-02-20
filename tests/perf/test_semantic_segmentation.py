"""OTX Semantic Segmentation perfomance tests."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest

from otx.cli.registry import Registry
from typing import Callable
from .benchmark import OTXBenchmark


MODEL_TEMPLATES = Registry(f"src/otx/algorithms").filter(task_type="SEGMENTATION").templates
MODEL_IDS = [template.model_template_id for template in MODEL_TEMPLATES]


class TestPerfSemanticSegmentation:
    """Benchmark basic semantic segmentation."""

    BENCHMARK_CONFIGS = {
        "small": {
            "tags": {
                "task": "semantic_segmentation",
            },
            "datasets": [
                "semantic_seg/kvasir_small/1",
                "semantic_seg/kvasir_small/2",
                "semantic_seg/kvasir_small/3",
            ],
            "subset_dir_names": {"train": "train", "val": "val", "test": "test"},
            "num_repeat": 3,
        },
        "medium": {
            "tags": {
                "task": "semantic_segmentation",
            },
            "datasets": [
                "semantic_seg/kvasir_medium",
            ],
            "subset_dir_names": {"train": "train", "val": "val", "test": "test"},
            "num_repeat": 3,
        },
        "large": {
            "tags": {
                "task": "semantic_segmentation",
            },
            "datasets": [
                "semantic_seg/kvasir_large",
            ],
            "subset_dir_names": {"train": "train", "val": "val", "test": "test"},
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
                    "name": "Dice Average(train)",
                    "op": ">",
                    "margin": 0.1,
                },
                {
                    "name": "epoch",
                    "op": "<",
                    "margin": 0.1,
                },
                {
                    "name": "Dice Average(export)",
                    "op": ">",
                    "margin": 0.1,
                },
                {
                    "name": "Dice Average(optimize)",
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
