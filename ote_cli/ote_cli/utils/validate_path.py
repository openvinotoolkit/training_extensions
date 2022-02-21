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

import os
import errno


def is_path_valid(test_string: str) -> bool:
    try:
        if not isinstance(test_string, str) or not test_string:
            return False
        _, test_string = os.path.splitdrive(test_string)

        root = os.path.sep
        assert os.path.isdir(root)

        root = root.rstrip(os.path.sep) + os.path.sep
        for pathname_part in test_string.split(os.path.sep):
            try:
                os.lstat(root + pathname_part)
            except OSError as exc:
                if exc.errno in {errno.ENAMETOOLONG, errno.ERANGE}:
                    return False
            except ValueError:
                return False
    except TypeError:
        return False
    else:
        return True


def validate_single_path(path: str) -> None:
    if not is_path_valid(path):
        raise Exception(f"Path is not valid: {path}")


def validate_path(test_string: str) -> None:
    if ',' in str(test_string):
        paths = test_string.split(',')
        for path in paths:
            validate_single_path(path)
    else:
        validate_single_path(test_string)
