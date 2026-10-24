# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Prevent pytest from collecting any files under tests/assets/.

The download.py scripts in dataset subdirectories are not test modules.
"""

collect_ignore_glob = ["*"]
