"""
The util for validation paths that sourced to parameters
"""
# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import errno
import os


def is_path_valid(test_string: str) -> bool:
    """
    Validate a string path for correctness

    """
    try:
        if not isinstance(test_string, str) or not test_string:
            return False
        if not all(s.isprintable() for s in test_string):
            return False

        try:
            os.lstat(test_string)
        except OSError as exc:
            if exc.errno in {errno.ENAMETOOLONG, errno.ERANGE}:
                return False
    except TypeError:
        return False
    else:
        return True


def validate_single_path(path: str) -> None:
    """
    Wrapper for single path
    """
    if not is_path_valid(path):
        raise Exception(f"Path is not valid: {path}")


def validate_path(test_string: str) -> None:
    """
    Wrapper for multiple paths
    """
    if ',' in str(test_string):
        paths = test_string.split(',')
        for path in paths:
            validate_single_path(path)
    else:
        validate_single_path(test_string)
