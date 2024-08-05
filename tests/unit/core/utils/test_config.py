# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from otx.core.utils.config import to_list, to_tuple


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
