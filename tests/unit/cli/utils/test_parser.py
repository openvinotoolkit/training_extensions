# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import argparse

import pytest

from otx.cli.utils.parser import MemSizeAction


@pytest.fixture
def fxt_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mem-cache-size",
        action=MemSizeAction,
        type=str,
        required=False,
        default=0,
    )
    return parser


@pytest.mark.parametrize(
    "mem_size_arg,expected",
    [
        ("1561", 1561),
        ("121k", 121 * (2**10)),
        ("121kb", 121 * (2**10)),
        ("121kib", 121 * (10**3)),
        ("121m", 121 * (2**20)),
        ("121mb", 121 * (2**20)),
        ("121mib", 121 * (10**6)),
        ("121g", 121 * (2**30)),
        ("121gb", 121 * (2**30)),
        ("121gib", 121 * (10**9)),
        ("121as", None),
        ("121dddd", None),
    ],
)
def test_mem_size_parsing(fxt_argparse, mem_size_arg, expected):
    try:
        args = fxt_argparse.parse_args(["--mem-cache-size", mem_size_arg])
        assert args.mem_cache_size == expected
    except ValueError:
        assert expected is None
