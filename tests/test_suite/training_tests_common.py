# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os.path as osp
from typing import Union

from otx.api.entities.metrics import Performance, ScoreMetric

# This string constant will be used as a special constant for a config field
# value to point that the field should be filled in tests' code by some default
# value specific for this field.
DEFAULT_FIELD_VALUE_FOR_USING_IN_TEST = "DEFAULT_FIELD_VALUE_FOR_USING_IN_TEST"


# This string constant will be used as a special constant for a config field value to point
# that the field should NOT be changed in tests -- its value should be taken
# from the template file or the config file of the model.
KEEP_CONFIG_FIELD_VALUE = "CONFIG"


# This is a constant for pointing usecase for reallife training tests
REALLIFE_USECASE_CONSTANT = "reallife"


# Constant for storing in dict-s with paths the root path
# that will be used for resolving relative paths.
ROOT_PATH_KEY = "_root_path"


def make_path_be_abs(some_val, root_path):
    assert isinstance(some_val, str), f"Wrong type of value: {some_val}, type={type(some_val)}"

    assert isinstance(root_path, str), f"Wrong type of root_path: {root_path}, type={type(root_path)}"

    # Note that os.path.join(a, b) == b if b is an absolute path
    return osp.join(root_path, some_val)


def make_paths_be_abs(some_dict, root_path):
    assert isinstance(some_dict, dict), f"Wrong type of value: {some_dict}, type={type(some_dict)}"

    assert isinstance(root_path, str), f"Wrong type of root_path: {root_path}, type={type(root_path)}"

    assert all(isinstance(v, str) for v in some_dict.values()), f"Wrong input dict {some_dict}"

    for k in list(some_dict.keys()):
        # Note that os.path.join(a, b) == b if b is an absolute path
        some_dict[k] = osp.join(root_path, some_dict[k])
    return some_dict


def performance_to_score_name_value(perf: Union[Performance, None]):
    """
    The method is intended to get main score info from Performance class
    """
    if perf is None:
        return None, None
    assert isinstance(perf, Performance)
    score = perf.score
    assert isinstance(score, ScoreMetric)
    assert isinstance(score.name, str) and score.name, f'Wrong score name "{score.name}"'
    return score.name, score.value
