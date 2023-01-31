# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from abc import ABC, abstractmethod
from collections import OrderedDict
from pprint import pformat
from typing import Any, Callable, Dict, Optional

import pytest

from .e2e_test_system import DataCollector
from .logging import get_logger
from .training_tests_actions import BaseOTXTestAction
from .training_tests_common import REALLIFE_USECASE_CONSTANT

logger = get_logger()


def get_value_from_dict_by_dot_separated_address(struct, address):
    def _get(cur_struct, addr):
        assert isinstance(addr, list)
        if not addr:
            return cur_struct
        assert isinstance(cur_struct, dict)
        if addr[0] not in cur_struct:
            raise ValueError(f"Cannot find address {address} in struct {struct}: {addr[0]} is absent in {cur_struct}")
        return _get(cur_struct[addr[0]], addr[1:])

    assert isinstance(address, str), f"The parameter address should be string, address={address}"
    return _get(struct, address.split("."))


class Validator:
    """
    The class receives info on results metric of the current test stage and
    compares it with the expected metrics.
    """

    def __init__(self, cur_test_expected_metrics_callback: Optional[Callable[[], Dict]]):
        self.cur_test_expected_metrics_callback = cur_test_expected_metrics_callback

    # TODO(lbeynens): add a method to extract dependency info from expected metrics
    #                 to add the stages we depend on to the dependency list.

    @staticmethod
    def _get_min_max_value_from_expected_metrics(cur_metric_requirements: Dict, test_results_storage: Dict):
        """
        The method gets requirement for some metric and convert it to the triplet
        (target_value, min_value, max_value).
        Note that the target_value may be pointed either by key 'target_value' (in this case it is float),
        or by the key 'base', in this case it is a dot-separated address to another value in the
        storage of previous stages' results `test_results_storage`.

        Note that the range for the metric values may be pointed by key 'max_diff',
        in this case the range will be [target_value - max_diff, target_value + max_diff]
        (inclusively).

        But also the range may be pointed by keys 'max_diff_if_less_threshold' and
        'max_diff_if_greater_threshold', in this case the range is
        [target_value - max_diff_if_less_threshold, target_value + max_diff_if_greater_threshold]
        (also inclusively). This allows to point non-symmetric ranges w.r.t. the target_value.

        Also note that if one of 'max_diff_if_less_threshold' and 'max_diff_if_greater_threshold'
        is absent, it is set to `+infinity`, so the range will be bounded from one side
        (but not both of them, this will be an error)
        """
        keys = set(cur_metric_requirements.keys())
        if "target_value" not in keys and "base" not in keys:
            raise ValueError(
                f'Wrong cur_metric_requirements: either "target_value" or "base" '
                f" should be pointed in the structure, whereas "
                f"cur_metric_requirements={pformat(cur_metric_requirements)}"
            )
        if "target_value" in keys and "base" in keys:
            raise ValueError(
                f'Wrong cur_metric_requirements: either "target_value" or "base" '
                f" should be pointed in the structure, but not both, whereas "
                f"cur_metric_requirements={pformat(cur_metric_requirements)}"
            )
        if (
            ("max_diff" not in keys)
            and ("max_diff_if_less_threshold" not in keys)
            and ("max_diff_if_greater_threshold" not in keys)
        ):
            raise ValueError(
                f'Wrong cur_metric_requirements: either "max_diff" or one/two of '
                f'"max_diff_if_less_threshold" and "max_diff_if_greater_threshold" should be '
                f"pointed in the structure, whereas "
                f"cur_metric_requirements={pformat(cur_metric_requirements)}"
            )

        if ("max_diff" in keys) and ("max_diff_if_less_threshold" in keys or "max_diff_if_greater_threshold" in keys):
            raise ValueError(
                f'Wrong cur_metric_requirements: either "max_diff" or one/two of '
                f'"max_diff_if_less_threshold" and "max_diff_if_greater_threshold" should be '
                f"pointed in the structure, but not both, whereas "
                f"cur_metric_requirements={pformat(cur_metric_requirements)}"
            )

        if "target_value" in cur_metric_requirements:
            target_value = float(cur_metric_requirements["target_value"])
        elif "base" in cur_metric_requirements:
            base_metric_address = cur_metric_requirements["base"]
            target_value = get_value_from_dict_by_dot_separated_address(test_results_storage, base_metric_address)
            target_value = float(target_value)
        else:
            raise RuntimeError(f"ERROR: Wrong parsing of metric requirements {cur_metric_requirements}")

        if "max_diff" in cur_metric_requirements:
            max_diff = cur_metric_requirements["max_diff"]
            max_diff = float(max_diff)
            if not max_diff >= 0:
                raise ValueError(f"Wrong max_diff {max_diff} -- it should be a non-negative number")
            return (target_value, target_value - max_diff, target_value + max_diff)

        max_diff_if_less_threshold = cur_metric_requirements.get("max_diff_if_less_threshold")
        max_diff_if_greater_threshold = cur_metric_requirements.get("max_diff_if_greater_threshold")
        if max_diff_if_less_threshold is None and max_diff_if_greater_threshold is None:
            raise ValueError(
                f"Wrong cur_metric_requirements: all of max_diff, max_diff_if_less_threshold, and "
                f"max_diff_if_greater_threshold are None, "
                f"cur_metric_requirements={pformat(cur_metric_requirements)}"
            )

        if max_diff_if_greater_threshold is not None:
            max_diff_if_greater_threshold = float(max_diff_if_greater_threshold)
            if not max_diff_if_greater_threshold >= 0:
                raise ValueError(
                    f"Wrong max_diff_if_greater_threshold {max_diff_if_greater_threshold} "
                    f"-- it should be a non-negative number"
                )

            max_value = target_value + max_diff_if_greater_threshold
        else:
            max_value = None

        if max_diff_if_less_threshold is not None:
            max_diff_if_less_threshold = float(max_diff_if_less_threshold)
            if not max_diff_if_less_threshold >= 0:
                raise ValueError(
                    f"Wrong max_diff_if_less_threshold {max_diff_if_less_threshold} "
                    f"-- it should be a non-negative number"
                )

            min_value = target_value - max_diff_if_less_threshold
        else:
            min_value = None

        return (target_value, min_value, max_value)

    @staticmethod
    def _compare(
        current_metric: float,
        cur_res_addr: str,
        target_value: float,
        min_value: Optional[float],
        max_value: Optional[float],
    ):
        assert all(isinstance(v, float) for v in [current_metric, target_value])
        assert all((v is None) or isinstance(v, float) for v in [min_value, max_value])

        if min_value is not None and max_value is not None:
            assert min_value <= target_value <= max_value

            if min_value <= current_metric <= max_value:
                logger.info(
                    f"Validation: passed: The metric {cur_res_addr} is in the acceptable range "
                    f"near the target value {target_value}: "
                    f"{current_metric} is in [{min_value}, {max_value}]"
                )
                is_passed = True
                cur_fail_reason = None
            else:
                cur_fail_reason = (
                    f"Validation: failed: The metric {cur_res_addr} is NOT in the acceptable range "
                    f"near the target value {target_value}: "
                    f"{current_metric} is NOT in [{min_value}, {max_value}]"
                )
                logger.error(cur_fail_reason)
                is_passed = False
            return is_passed, cur_fail_reason

        assert (min_value is not None) or (max_value is not None)
        if min_value is not None:
            cmp_op = lambda x: x >= min_value  # noqa: E731
            cmp_str_true = "greater or equal"
            cmp_op_str_true = ">="
            cmp_op_str_false = "<"
            threshold = min_value
        else:
            assert max_value is not None
            cmp_op = lambda x: x <= max_value  # noqa: E731
            cmp_str_true = "less or equal"
            cmp_op_str_true = "<="
            cmp_op_str_false = ">"
            threshold = max_value
        acceptable_error = abs(threshold - target_value)
        if cmp_op(current_metric):
            logger.info(
                f"Validation: passed: The metric {cur_res_addr} is {cmp_str_true} "
                f"the target value {target_value} with acceptable error {acceptable_error}: "
                f"{current_metric} {cmp_op_str_true} {threshold}"
            )
            is_passed = True
            cur_fail_reason = None
        else:
            cur_fail_reason = (
                f"Validation: failed: The metric {cur_res_addr} is NOT {cmp_str_true} "
                f"the target value {target_value} with acceptable error {acceptable_error}: "
                f"{current_metric} {cmp_op_str_false} {threshold}"
            )
            logger.error(cur_fail_reason)
            is_passed = False
        return is_passed, cur_fail_reason

    def validate(self, current_result: Dict, test_results_storage: Dict):
        """
        The method validates results of the current test.
        :param current_result -- dict with result of the current test
        :param test_results_storage -- dict with results of previous tests
                                       of this test case
                                       (e.g. the same training parameters)

        The function returns nothing, but may raise exceptions to fail the test.
        If the structure stored expected metrics is wrong, the function raises ValueError.
        """
        if self.cur_test_expected_metrics_callback is None:
            # most probably, it is not a reallife test
            logger.info(
                f"Validation: skipped, since there should not be expected metrics for this test, "
                f'most probably the test is not run in "{REALLIFE_USECASE_CONSTANT}" usecase'
            )
            return

        logger.info("Validation: begin")

        # calling the callback to receive expected metrics for the current test
        cur_test_expected_metrics = self.cur_test_expected_metrics_callback()

        assert isinstance(
            cur_test_expected_metrics, dict
        ), f"Wrong current test expected metric: {cur_test_expected_metrics}"
        logger.debug(f"Validation: received cur_test_expected_metrics={pformat(cur_test_expected_metrics)}")
        is_passed = True
        fail_reasons = []
        for k, v in cur_test_expected_metrics.items():
            # TODO(lbeynens): add possibility to point a list of requirements for a metric
            cur_res_addr = k
            cur_metric_requirements = v
            logger.info(f"Validation: begin check {cur_res_addr}")
            try:
                current_metric = get_value_from_dict_by_dot_separated_address(current_result, cur_res_addr)
                current_metric = float(current_metric)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Cannot get metric {cur_res_addr} from the current result {current_result}") from e

            logger.debug(f"current_metric = {current_metric}")
            try:
                (
                    target_value,
                    min_value,
                    max_value,
                ) = self._get_min_max_value_from_expected_metrics(cur_metric_requirements, test_results_storage)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Error when parsing expected metrics for the metric {cur_res_addr}") from e

            cur_is_passed, cur_fail_reason = self._compare(
                current_metric, cur_res_addr, target_value, min_value, max_value
            )
            if not cur_is_passed:
                is_passed = False
                fail_reasons.append(cur_fail_reason)

            logger.info(f"Validation: end check {cur_res_addr}")

        logger.info(f"Validation: end, result={is_passed}")
        if not is_passed:
            fail_reasons_str = "\n".join(fail_reasons)
            pytest.fail(f"Validation failed:\n{fail_reasons_str}")


class OTXTestStagesStorageInterface(ABC):
    @abstractmethod
    def get_stage(self, name: str) -> "OTXTestStage":
        raise NotImplementedError("The method get_stage is not implemented")


class OTXTestStage:
    """
    OTXTestStage -- auxiliary class that
    1. Allows to set up dependency between test stages: before the main action of a test stage is run, all the actions
       for the stages that are pointed in 'depends' list are called beforehand;
    2. Runs for each test stage its main action only once: the main action is run inside try-except clause, and
       2.1. if the action was executed without exceptions, a flag `was_processed` is set, the results of the action
            are kept, and the next time the stage is called no action is executed;
       2.2. if the action raised an exception, the exception is stored, the flag `was_processed` is set, and the next
            time the stage is called the exception is re-raised.
    """

    def __init__(self, action: BaseOTXTestAction, stages_storage: OTXTestStagesStorageInterface):
        self.was_processed = False
        self.stored_exception: Optional[Exception] = None
        self.action = action
        self.stages_storage = stages_storage
        self.stage_results: Dict[str, Any] = {}
        assert isinstance(self.stages_storage, OTXTestStagesStorageInterface)
        assert isinstance(self.action, BaseOTXTestAction)

    def __str__(self):
        return (
            f"{type(self).__name__}("
            f"action={self.action}, "
            f"was_processed={self.was_processed}, "
            f"stored_exception={self.stored_exception}, "
            f"stage_results.keys={list(self.stage_results.keys())}, "
            f"id(stages_storage)={id(self.stages_storage)}"
            f")"
        )

    @property
    def name(self):
        return self.action.name

    def get_depends_stages(self):
        logger.debug(f"get_depends_stages for stage {self.name}: begin")
        depends_stages_names = self.action.depends_stages_names
        assert isinstance(depends_stages_names, list)
        assert all(isinstance(v, str) for v in depends_stages_names)

        stages = []
        for stage_name in depends_stages_names:
            logger.debug(f'get_depends_stages: get stage with name "{stage_name}"')
            cur_stage = self.stages_storage.get_stage(stage_name)
            assert isinstance(cur_stage, OTXTestStage), f'Wrong stage for stage_name="{stage_name}"'
            assert (
                cur_stage.name == stage_name
            ), f'For stage_name="{stage_name}" got the stage with name="{cur_stage.name}"'
            logger.debug(f'get_depends_stages: cur_stage="{cur_stage}"')
            stages.append(cur_stage)
        logger.debug(f"get_depends_stages for stage {self.name}: end")
        return stages

    def _reraise_stage_exception_if_was_failed(self):
        assert (
            self.was_processed
        ), "The method _reraise_stage_exception_if_was_failed should be used only for stages that were processed"
        if self.stored_exception is None:
            # nothing to do here
            return

        logger.warning(
            f"In stage {self.name}: found that previous call of the stage " "caused exception -- re-raising it"
        )
        raise self.stored_exception

    def _run_validation(self, test_results_storage: Dict, validator: Optional[Validator]):
        if not self.action.with_validation:
            return
        if validator is None:
            logger.debug(
                "The validator is None -- the validation should be skipped, "
                "most probably this test stage was run from a dependency chain"
            )
            return

        validator.validate(self.stage_results, test_results_storage)

    def run_once(
        self,
        data_collector: DataCollector,
        test_results_storage: OrderedDict,
        validator: Optional[Validator],
    ):
        logger.info(f'Begin stage "{self.name}"')
        assert isinstance(test_results_storage, OrderedDict)
        logger.debug(f'For test stage "{self.name}": test_results_storage.keys = {list(test_results_storage.keys())}')

        for dep_stage in self.get_depends_stages():
            # Processing all dependency stages of the current test.
            # Note that
            # * the stages may run their own dependency stages -- they will compose so called "dependency chain"
            # * the dependency stages are run with `validator = None`
            #   to avoid validation of stages that are run from the dependency chain.
            logger.debug(f'For test stage "{self.name}": Before running dep. stage "{dep_stage.name}"')
            dep_stage.run_once(data_collector, test_results_storage, validator=None)
            logger.debug(f'For test stage "{self.name}": After running dep. stage "{dep_stage.name}"')

        if self.was_processed:
            self._reraise_stage_exception_if_was_failed()
            # if we are here, then the stage was processed without exceptions
            logger.info(f"The stage {self.name} was already processed SUCCESSFULLY")

            # Run validation here for the rare case if this test now is being run *not* from a dependency chain
            # (i.e. the test is run with `validator != None`),
            # but the test already has been run from some dependency chain earlier.
            self._run_validation(test_results_storage, validator)

            logger.info(f'End stage "{self.name}"')
            return

        if self.name in test_results_storage:
            raise RuntimeError(
                f'Error: For test stage "{self.name}": '
                f"another OTXTestStage with name {self.name} has been run already"
            )

        try:
            logger.info(f'For test stage "{self.name}": Before running main action')
            self.stage_results = self.action(data_collector=data_collector, results_prev_stages=test_results_storage)
            logger.info(f'For test stage "{self.name}": After running main action')
            self.was_processed = True
            test_results_storage[self.name] = self.stage_results
            logger.debug(
                f'For test stage "{self.name}": after addition test_results_storage.keys = '
                f"{list(test_results_storage.keys())}"
            )
        except Exception as e:
            logger.info(
                f'For test stage "{self.name}": After running action for stage {self.name} -- CAUGHT EXCEPTION:\n{e}'
            )
            logger.info(f'End stage "{self.name}"')
            self.stored_exception = e
            self.was_processed = True
            raise e

        # The validation step is made outside the central try...except clause, since if the test was successful, but
        # the quality numbers were lower than expected, the result of the stage still may be re-used
        # in other stages.
        self._run_validation(test_results_storage, validator)
        logger.info(f'End stage "{self.name}"')
