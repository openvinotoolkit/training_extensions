# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from omegaconf import OmegaConf
from otx.core.utils.config import inplace_num_classes, to_list, to_tuple


def test_to_tuple() -> None:
    input_dict = {
        "a": [1, 2, 3],
        "b": {
            "c": (4, 5),
            "d": [6, 7, 8],
        },
        "e": {
            "f": {
                "g": [9, 10],
            },
        },
    }

    expected_output = {
        "a": (1, 2, 3),
        "b": {
            "c": (4, 5),
            "d": (6, 7, 8),
        },
        "e": {
            "f": {
                "g": (9, 10),
            },
        },
    }

    assert to_tuple(input_dict) == expected_output


def test_to_list() -> None:
    input_dict = {}
    expected_output = {}
    assert to_list(input_dict) == expected_output

    input_dict = {"a": (1, 2, 3), "b": {"c": (4, 5)}}
    expected_output = {"a": [1, 2, 3], "b": {"c": [4, 5]}}
    assert to_list(input_dict) == expected_output

    input_dict = {"a": [1, 2, 3], "b": {"c": [4, 5]}}
    expected_output = {"a": [1, 2, 3], "b": {"c": [4, 5]}}
    assert to_list(input_dict) == expected_output

    input_dict = {"a": (1, 2, 3), "b": [4, 5]}
    expected_output = {"a": [1, 2, 3], "b": [4, 5]}
    assert to_list(input_dict) == expected_output

    input_dict = {"a": {"b": (1, 2, 3)}, "c": {"d": {"e": (4, 5)}}}
    expected_output = {"a": {"b": [1, 2, 3]}, "c": {"d": {"e": [4, 5]}}}
    assert to_list(input_dict) == expected_output

    input_dict = {"a": {"b": [1, 2, 3]}, "c": {"d": {"e": [4, 5]}}}
    expected_output = {"a": {"b": [1, 2, 3]}, "c": {"d": {"e": [4, 5]}}}
    assert to_list(input_dict) == expected_output

    input_dict = {"a": {"b": (1, 2, 3)}, "c": {"d": [4, 5]}}
    expected_output = {"a": {"b": [1, 2, 3]}, "c": {"d": [4, 5]}}
    assert to_list(input_dict) == expected_output


def test_inplace_num_classes() -> None:
    cfg = OmegaConf.create({"num_classes": 10, "model": {"num_classes": 5}})
    inplace_num_classes(cfg, 20)
    assert cfg.num_classes == 20
    assert cfg.model.num_classes == 20

    cfg = OmegaConf.create([{"num_classes": 10}, {"num_classes": 5}])
    inplace_num_classes(cfg, 20)
    assert cfg[0].num_classes == 20
    assert cfg[1].num_classes == 20

    cfg = OmegaConf.create({"model": {"num_classes": 10, "layers": [{"num_classes": 5}]}})
    inplace_num_classes(cfg, 20)
    assert cfg.model.num_classes == 20
    assert cfg.model.layers[0].num_classes == 20
