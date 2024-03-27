"""OTX Instance Segmentation perfomance tests."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest

from otx.cli.registry import Registry
from typing import Callable
from .benchmark import Benchmark


MODEL_TEMPLATES = Registry(f"src/otx/algorithms").filter(task_type="INSTANCE_SEGMENTATION").templates
MODEL_IDS = [template.model_template_id for template in MODEL_TEMPLATES]


class TestPerfInstanceSegmentation:
    """Benchmark basic instance segmentation."""

    BENCHMARK_CONFIGS = {
        "small": {
            "tags": {
                "task": "instance_segmentation",
            },
            "datasets": [
                "instance_seg/wgisd_small/1",
                "instance_seg/wgisd_small/2",
                "instance_seg/wgisd_small/3",
            ],
            "num_repeat": 5,
        },
        "medium": {
            "tags": {
                "task": "instance_segmentation",
            },
            "datasets": [
                "instance_seg/coco_car_person_medium",
            ],
            "num_repeat": 5,
        },
        # TODO: Refine large dataset
        # "large": {
        #     "tags": {
        #         "task": "instance_segmentation",
        #     },
        #     "datasets": [
        #         "instance_seg/bdd_large",
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


class TestPerfTilingInstanceSegmentation:
    """Benchmark tiling instance segmentation."""

    TILING_PARAMS = {
        "tiling_parameters.enable_tiling": 1,
    }
    BENCHMARK_CONFIGS = {
        "small": {
            "tags": {
                "task": "tiling_instance_segmentation",
            },
            "datasets": [
                "tiling_instance_seg/vitens_aeromonas_small/1",
                "tiling_instance_seg/vitens_aeromonas_small/2",
                "tiling_instance_seg/vitens_aeromonas_small/3",
            ],
            "num_repeat": 5,
            "train_params": TILING_PARAMS,
        },
        "medium": {
            "tags": {
                "task": "tiling_instance_segmentation",
            },
            "datasets": [
                "tiling_instance_seg/vitens_aeromonas_medium",
            ],
            "num_repeat": 5,
            "train_params": TILING_PARAMS,
        },
        # TODO: Refine large dataset
        # "large": {
        #     "tags": {
        #         "task": "tiling_instance_segmentation",
        #     },
        #     "datasets": [
        #         "tiling_instance_seg/dota_large",
        #     ],
        #     "num_repeat": 5,
        #     "train_params": TILING_PARAMS,
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
