# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import json
import os
import tempfile

import pytest

from otx.algorithms.common.adapters.nncf.config import (
    compose_nncf_config,
    load_nncf_config,
    merge_dicts_and_lists_b_into_a,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
def test_load_nncf_config():
    with pytest.raises(AssertionError):
        load_nncf_config("invalid_path")
    with tempfile.TemporaryDirectory() as directory:
        tmp = os.path.join(directory, "temp.json")
        with open(tmp, "w") as f:
            json.dump({"dummy": "dummy"}, f)
        assert {"dummy": "dummy"} == load_nncf_config(tmp)


@e2e_pytest_unit
def test_compose_nncf_config():
    nncf_config = {
        "base": {
            "find_unused_parameters": True,
            "nncf_config": {
                "target_metric_name": "mAP",
                "input_info": {"sample_size": [1, 3, 864, 864]},
                "compression": [],
                "log_dir": "/tmp",
                "accuracy_aware_training": {
                    "mode": "early_exit",
                },
            },
        },
        "nncf_quantization": {
            "optimizer": {"lr": 0.0005},
            "nncf_config": {
                "compression": [
                    {
                        "algorithm": "quantization",
                        "initializer": {
                            "range": {"num_init_samples": 300},
                            "batchnorm_adaptation": {"num_bn_adaptation_samples": 300},
                        },
                    }
                ],
                "accuracy_aware_training": {
                    "mode": "early_exit",
                    "params": {
                        "maximal_absolute_accuracy_degradation": 0.01,
                        "maximal_total_epochs": 20,
                    },
                },
            },
        },
        "nncf_quantization_pruning": {
            "nncf_config": {
                "accuracy_aware_training": {
                    "mode": "adaptive_compression_level",
                    "params": {
                        "initial_training_phase_epochs": 5,
                        "maximal_total_epochs": 100,
                        "patience_epochs": 5,
                    },
                },
                "compression": [
                    {
                        "algorithm": "filter_pruning",
                        "ignored_scopes": ["{re}SingleStageDetector/SSDHead\\[bbox_head\\].*"],
                        "params": {
                            "schedule": "baseline",
                            "pruning_flops_target": 0.1,
                            "filter_importance": "geometric_median",
                        },
                    },
                    {
                        "algorithm": "quantization",
                        "initializer": {
                            "range": {"num_init_samples": 300},
                            "batchnorm_adaptation": {"num_bn_adaptation_samples": 300},
                        },
                    },
                ],
            }
        },
        "error": {"nncf_config": {"accuracy_aware_training": ["error"]}},
        "order_of_parts": ["nncf_quantization", "nncf_quantization_pruning", "error"],
    }

    assert nncf_config["base"] == compose_nncf_config(nncf_config, [])
    assert merge_dicts_and_lists_b_into_a(nncf_config["base"], nncf_config["nncf_quantization"]) == compose_nncf_config(
        nncf_config, ["nncf_quantization"]
    )
    assert merge_dicts_and_lists_b_into_a(
        nncf_config["base"], nncf_config["nncf_quantization_pruning"]
    ) == compose_nncf_config(nncf_config, ["nncf_quantization_pruning"])

    with pytest.raises(RuntimeError):
        compose_nncf_config(nncf_config, ["error"])


@e2e_pytest_unit
def test_merge_dicts_and_lists_b_into_a():
    assert {"a": 1, "b": 2} == merge_dicts_and_lists_b_into_a({"a": 1}, {"b": 2})
    assert [1, 2] == merge_dicts_and_lists_b_into_a([1], [2])
    assert {"a": [1, 2]} == merge_dicts_and_lists_b_into_a({"a": [1]}, {"a": [2]})
