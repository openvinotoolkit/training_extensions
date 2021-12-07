# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Union

from ote_sdk.entities.metrics import Performance, ScoreMetric


def DEFAULT_FIELD_VALUE_FOR_USING_IN_TEST():
    """
    This string constant will be used as a special constant for a config field
    value to point that the field should be filled in tests' code by some default
    value specific for this field.
    """
    return "DEFAULT_FIELD_VALUE_FOR_USING_IN_TEST"


def KEEP_CONFIG_FIELD_VALUE():
    """
    This string constant will be used as a special constant for a config field value to point
    that the field should NOT be changed in tests -- its value should be taken
    from the template file or the config file of the model.
    """
    return "KEEP_CONFIG_FIELD_VALUE"


def REALLIFE_USECASE_CONSTANT():
    """
    This is a constant for pointing usecase for reallife training tests
    """
    return "reallife"


def performance_to_score_name_value(perf: Union[Performance, None]):
    """
    The method is intended to get main score info from Performance class
    """
    if perf is None:
        return None, None
    assert isinstance(perf, Performance)
    score = perf.score
    assert isinstance(score, ScoreMetric)
    assert (
        isinstance(score.name, str) and score.name
    ), f'Wrong score name "{score.name}"'
    return score.name, score.value


def convert_hyperparams_to_dict(hyperparams):
    def _convert(p):
        if p is None:
            return None
        d = {}
        groups = getattr(p, "groups", [])
        parameters = getattr(p, "parameters", [])
        assert (not groups) or isinstance(
            groups, list
        ), f"Wrong field 'groups' of p={p}"
        assert (not parameters) or isinstance(
            parameters, list
        ), f"Wrong field 'parameters' of p={p}"
        for group_name in groups:
            g = getattr(p, group_name, None)
            d[group_name] = _convert(g)
        for par_name in parameters:
            d[par_name] = getattr(p, par_name, None)
        return d

    return _convert(hyperparams)
