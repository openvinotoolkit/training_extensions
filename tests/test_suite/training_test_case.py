# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from abc import abstractmethod
from collections import Counter, OrderedDict
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Type

from .e2e_test_system import DataCollector
from .logging import get_logger
from .training_tests_actions import BaseOTXTestAction
from .training_tests_stage import OTXTestStage, OTXTestStagesStorageInterface, Validator

logger = get_logger()


def _get_duplications(arr):
    c = Counter(arr)
    dups = [k for k, v in c.items() if v > 1]
    return dups


def _str_dict_with_shortened_vals(d, max_len=200):
    assert isinstance(d, dict)
    if not d:
        return "{}"

    def _shorten(v):
        sv = str(v)
        if len(sv) <= max_len:
            return sv
        return sv[:max_len] + "..."

    s = "\n".join(f"{k}: {_shorten(v)}," for k, v in d.items())
    s = "\n    ".join(s.split("\n"))
    s = "{\n    " + s + "\n}"
    return s


class OTXTestCaseInterface(OTXTestStagesStorageInterface):
    @classmethod
    @abstractmethod
    def get_list_of_test_stages(cls):
        raise NotImplementedError("The method get_list_of_test_stages is not implemented")

    @abstractmethod
    def run_stage(
        self,
        stage_name: str,
        data_collector: DataCollector,
        cur_test_expected_metrics_callback: Optional[Callable[[], Dict]],
    ):
        raise NotImplementedError("The method run_stage is not implemented")


def generate_otx_integration_test_case_class(
    test_actions_classes: List[Type[BaseOTXTestAction]],
) -> Type:
    test_actions_classes = deepcopy(test_actions_classes)

    # check names' duplication
    classes_names = [action_cls._name for action_cls in test_actions_classes]
    name_dups = _get_duplications(classes_names)
    if name_dups:
        raise ValueError(f"Wrong input: there are duplications in names of actions; duplications = {name_dups}")

    class _OTXIntegrationTestCase(OTXTestCaseInterface):
        _TEST_STAGES = [action_cls._name for action_cls in test_actions_classes]

        @classmethod
        def get_list_of_test_stages(cls):
            return deepcopy(cls._TEST_STAGES)

        def __init__(self, params_factories_for_test_actions: Dict[str, Callable[[], Dict]]):
            logger.debug("initialization of test case: begin")
            self._stages = OrderedDict()
            for action_cls in test_actions_classes:
                logger.debug(f"initialization of test case: action_cls={action_cls}")

                cur_name = action_cls._name
                assert cur_name is not None
                cur_params_factory = params_factories_for_test_actions.get(cur_name)
                if cur_params_factory is not None:
                    logger.debug("initialization of test case: calling params factory")
                    cur_params = cur_params_factory()
                else:
                    cur_params = {}

                assert isinstance(cur_params, dict), f"Wrong params received from factory: {cur_params}"
                short_params_str = _str_dict_with_shortened_vals(cur_params)
                logger.info(f"initialization of test case: add action '{cur_name}' " f"with params={short_params_str}")

                cur_action = action_cls(**cur_params)

                # Note that `self` is used as stages_storage for OTXTestStage below
                cur_stage = OTXTestStage(action=cur_action, stages_storage=self)
                self._stages[cur_name] = cur_stage

            assert list(self._stages.keys()) == list(self._TEST_STAGES)

            # test results should be kept between stages
            self.test_results_storage: OrderedDict = OrderedDict()
            logger.debug("initialization of test case: end")

        # implementation of method from OTXTestStagesStorageInterface
        def get_stage(self, name: str) -> "OTXTestStage":
            return self._stages[name]

        def run_stage(
            self,
            stage_name: str,
            data_collector: DataCollector,
            cur_test_expected_metrics_callback: Optional[Callable[[], Dict]],
        ):
            assert stage_name in self._TEST_STAGES, f"Wrong stage_name {stage_name}"
            validator = Validator(cur_test_expected_metrics_callback)
            self._stages[stage_name].run_once(data_collector, self.test_results_storage, validator)

    return _OTXIntegrationTestCase
