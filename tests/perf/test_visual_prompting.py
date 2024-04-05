"""OTX Visual Prompting perfomance tests."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest

from otx.cli.registry import Registry
from typing import Callable
from .benchmark import Benchmark


class TestPerfVisualPrompting:
    """Benchmark visual prompting."""

    MODEL_TEMPLATES = [
        template
        for template in Registry("src/otx/algorithms/visual_prompting").filter(task_type="VISUAL_PROMPTING").templates
        if "Zero_Shot" not in template.name
    ]
    MODEL_IDS = [template.model_template_id for template in MODEL_TEMPLATES]

    BENCHMARK_CONFIGS = {
        "small": {
            "tags": {
                "task": "visual_prompting",
            },
            "datasets": [
                "visual_prompting/wgisd_small/1",
                "visual_prompting/wgisd_small/2",
                "visual_prompting/wgisd_small/3",
            ],
            "num_repeat": 5,
        },
        "medium": {
            "tags": {
                "task": "visual_prompting",
            },
            "datasets": [
                "visual_prompting/coco_car_person_medium",
            ],
            "num_repeat": 5,
        },
        "large": {
            "tags": {
                "task": "visual_prompting",
            },
            "datasets": [
                "visual_prompting/Vitens-Coliform-coco",
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
                    "name": "Dice Average(train)",
                    "op": ">",
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


class TestPerfZeroShotVisualPrompting:
    """Benchmark zero-shot visual prompting."""

    MODEL_TEMPLATES = [
        template
        for template in Registry("src/otx/algorithms/visual_prompting").filter(task_type="VISUAL_PROMPTING").templates
        if "Zero_Shot" in template.name
    ]
    MODEL_IDS = [template.model_template_id for template in MODEL_TEMPLATES]

    BENCHMARK_CONFIGS = {
        "medium": {
            "tags": {
                "task": "zero_shot_visual_prompting",
            },
            "datasets": [
                "zero_shot_visual_prompting/coco_car_person_medium",
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
