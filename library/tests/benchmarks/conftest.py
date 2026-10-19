# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


def pytest_addoption(parser):
    parser.addoption(
        "--data_root",
        action="store",
        default="data",
        help="Root directory for test data",
    )
