"""OTX Instance Segmentation perfomance tests."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest

from otx.cli.registry import Registry
from .benchmark import OTXBenchmark


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
            "num_repeat": 3,
        },
        "medium": {
            "tags": {
                "task": "instance_segmentation",
            },
            "datasets": [
                "instance_seg/coco_car_person_medium",
            ],
            "num_repeat": 3,
        },
        "large": {
            "tags": {
                "task": "instance_segmentation",
            },
            "datasets": [
                "instance_seg/bdd_large",
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
            "num_repeat": 3,
            "train_params": TILING_PARAMS,
        },
        "medium": {
            "tags": {
                "task": "tiling_instance_segmentation",
            },
            "datasets": [
                "tiling_instance_seg/vitens_aeromonas_medium",
            ],
            "num_repeat": 3,
            "train_params": TILING_PARAMS,
        },
        "large": {
            "tags": {
                "task": "tiling_instance_segmentation",
            },
            "datasets": [
                "tiling_instance_seg/bdd_large",
            ],
            "num_repeat": 1,
            "train_params": TILING_PARAMS,
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
