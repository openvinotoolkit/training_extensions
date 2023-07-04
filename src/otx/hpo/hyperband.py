"""Hyperband implementation."""

# Copyright (C) 2022 Intel Corporation
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

import json
import logging
import math
import os
from os import path as osp
from typing import Any, Dict, List, Literal, Optional, Union

from scipy.stats.qmc import LatinHypercube

from otx.hpo.hpo_base import HpoBase, Trial, TrialStatus
from otx.hpo.utils import (
    check_mode_input,
    check_not_negative,
    check_positive,
    left_vlaue_is_better,
)

logger = logging.getLogger(__name__)


def _check_reduction_factor_value(reduction_factor: int):
    if reduction_factor < 2:
        raise ValueError("reduction_factor should be greater than 2.\n" f"your value : {reduction_factor}")


class AshaTrial(Trial):
    """ASHA trial class.

    Args:
        trial_id (Any): Id of the trial.
        configuration (Dict): Configuration for the trial.
        train_environment (Optional[Dict]): Train environment for the trial.
                                            For, example, subset ratio can be included. Defaults to None.
    """

    def __init__(self, trial_id: Any, configuration: Dict, train_environment: Optional[Dict] = None):
        super().__init__(trial_id, configuration, train_environment)
        self._rung: Optional[int] = None
        self._bracket: Optional[int] = None
        self.estimating_max_resource: bool = False

    @property
    def rung(self):
        """Rung where the trial is included."""
        return self._rung

    @rung.setter
    def rung(self, val: int):
        """Setter for rung."""
        check_not_negative(val, "rung")
        self._rung = val

    @property
    def bracket(self):
        """Bracket where the trial is inlcuded."""
        return self._bracket

    @bracket.setter
    def bracket(self, val: int):
        """Setter for bracket."""
        check_not_negative(val, "bracket")
        self._bracket = val

    def save_results(self, save_path: str):
        """Save a result of the trial at 'save_path'."""
        results = {
            "id": self.id,
            "rung": self.rung,
            "configuration": self.configuration,
            "train_environment": self.train_environment,
            "score": self.score,
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f)


class Rung:
    """Rung class.

    Rung is in charge of selecting a trial to train and deciding which trial to promote to next rung in the bracket.

    Args:
        resource (Union[int, float]): Resource to use for training a trial.
                                      For example, something like epoch or iteration.
        num_required_trial (int): Necessary trials for the rung.
        reduction_factor (int): Decicdes how many trials to promote.
                                Only top 1 / reduction_factor of all trials can be promoted.
        rung_idx (int): Current rung index.
    """

    def __init__(
        self,
        resource: Union[int, float],
        num_required_trial: int,
        reduction_factor: int,
        rung_idx: int,
    ):
        check_positive(resource, "resource")
        check_positive(num_required_trial, "num_required_trial")
        _check_reduction_factor_value(reduction_factor)
        check_not_negative(rung_idx, "rung_idx")

        self._reduction_factor = reduction_factor
        self._num_required_trial = num_required_trial
        self._resource = resource
        self._trials: List[AshaTrial] = []
        self._rung_idx = rung_idx

    @property
    def num_required_trial(self):
        """Number of required trials for the rung."""
        return self._num_required_trial

    @property
    def resource(self):
        """Resource to use for training a trial."""
        return self._resource

    @property
    def rung_idx(self):
        """Current rung index."""
        return self._rung_idx

    def add_new_trial(self, trial: AshaTrial):
        """Add a new trial to the rung.

        Args:
            trial (AshaTrial): Trial to add

        Raises:
            RuntimeError: If no more trial is needed, raise an error.
        """
        if not self.need_more_trials():
            raise RuntimeError(f"{self.rung_idx} rung has already sufficient trials.")
        trial.iteration = self.resource
        trial.rung = self.rung_idx
        trial.status = TrialStatus.READY
        self._trials.append(trial)

    def get_best_trial(self, mode: str = "max") -> Optional[AshaTrial]:
        """Get best trial in the rung.

        Args:
            mode (str, optional): Decide which trial is better between having highest score or lowest score.
                                  Defaults to "max".

        Returns:
            Optional[AshaTrial]: Best trial. If there is no trial, return None.
        """
        check_mode_input(mode)
        best_score = None
        best_trial = None
        for trial in self._trials:
            if trial.rung != self.rung_idx:
                continue
            trial_score = trial.get_best_score(mode, self.resource)
            if trial_score is not None and (best_score is None or left_vlaue_is_better(trial_score, best_score, mode)):
                best_trial = trial
                best_score = trial_score

        return best_trial

    def need_more_trials(self) -> bool:
        """Check whether the rung needs more trials."""
        return self.num_required_trial > self.get_num_trials()

    def get_num_trials(self) -> int:
        """Number of trials the rung has."""
        return len(self._trials)

    def is_done(self) -> bool:
        """Check that the rung is done."""
        if self.need_more_trials():
            return False
        for trial in self._trials:
            if not trial.is_done():
                return False
        return True

    def get_trial_to_promote(self, asynchronous_sha: bool = False, mode: str = "max") -> Optional[AshaTrial]:
        """Get a trial to promote.

        Args:
            asynchronous_sha (bool, optional): Whether to operate SHA asynchronously. Defaults to False.
            mode (str, optional): Decide which trial is better between having highest score or lowest score.
                                  Defaults to "max".

        Returns:
            Optional[AshaTrial]: Trial to prmote. If there is no trial to promote, return None.
        """
        num_finished_trial = 0
        num_promoted_trial = 0
        best_score = None
        best_trial = None

        for trial in self._trials:
            if trial.rung == self._rung_idx:
                if trial.is_done() and trial.status != TrialStatus.RUNNING:
                    num_finished_trial += 1
                    trial_score = trial.get_best_score(mode, self.resource)
                    if best_score is None or left_vlaue_is_better(trial_score, best_score, mode):
                        best_trial = trial
                        best_score = trial_score
            else:
                num_promoted_trial += 1

        if asynchronous_sha:
            if (num_promoted_trial + num_finished_trial) // self._reduction_factor > num_promoted_trial:
                return best_trial
        elif self.is_done() and self._num_required_trial // self._reduction_factor > num_promoted_trial:
            return best_trial

        return None

    def get_next_trial(self) -> Optional[AshaTrial]:
        """Get next trial to trian.

        Returns:
            Optional[AshaTrial]: Next trial to train. If there is no left trial to train, then return None.
        """
        for trial in self._trials:
            if not trial.is_done() and trial.status != TrialStatus.RUNNING:
                return trial
        return None


class Bracket:
    """Bracket class. It operates a single SHA using multiple rungs.

    Args:
        bracket_id (int): Bracket id.
        minimum_resource (Union[float, int]): Maximum resource to use for training a trial.
        maximum_resource (Union[float, int]): Minimum resource to use for training a trial.
        hyper_parameter_configurations (List[AshaTrial]): Hyper parameter configuration to try.
        reduction_factor (int): Decicdes how many trials to promote to next rung.
                                Only top 1 / reduction_factor of rung trials can be promoted.
        mode (str, optional): Decide which trial is better between having highest score or lowest score.
                              Defaults to "max".
        asynchronous_sha (bool, optional): Whether to operate SHA asynchronously. Defaults to True.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        bracket_id: int,
        minimum_resource: Union[float, int],
        maximum_resource: Union[float, int],
        hyper_parameter_configurations: List[AshaTrial],
        reduction_factor: int = 3,
        mode: str = "max",
        asynchronous_sha: bool = True,
    ):
        # pylint: disable=too-many-arguments
        check_positive(minimum_resource, "minimum_resource")
        _check_reduction_factor_value(reduction_factor)
        check_mode_input(mode)

        self._id = bracket_id
        self._minimum_resource = minimum_resource
        self.maximum_resource = maximum_resource
        self._reduction_factor = reduction_factor
        self._mode = mode
        self._asynchronous_sha = asynchronous_sha
        self._trials: Dict[int, AshaTrial] = {}
        self._rungs: List[Rung] = self._initialize_rungs(hyper_parameter_configurations)

    @property
    def id(self):
        """Bracket id."""
        return self._id

    @property
    def maximum_resource(self):
        """Maximum resource to use for training a trial."""
        return self._maximum_resource

    @maximum_resource.setter
    def maximum_resource(self, val: Union[float, int]):
        """Setter for maximum_resource."""
        check_positive(val, "maximum_resource")
        if val < self._minimum_resource:
            raise ValueError(
                "maxnum_resource should be greater than minimum_resource.\n"
                f"value to set : {val}, minimum_resource : {self._minimum_resource}"
            )
        if val == self._minimum_resource:
            logger.warning("maximum_resource is same with the minimum_resource.")

        self._maximum_resource = val

    @property
    def max_rung(self):
        """Number of rungs the bracket has."""
        return self.calcuate_max_rung_idx(self._minimum_resource, self.maximum_resource, self._reduction_factor)

    @staticmethod
    def calcuate_max_rung_idx(
        minimum_resource: Union[float, int], maximum_resource: Union[float, int], reduction_factor: int
    ) -> int:
        """Calculate the number of rungs the bracket needs.

        Args:
            minimum_resource (Union[float, int]): Minimum resource to use for training a trial.
            maximum_resource (Union[float, int]): Maximum resource to use for training a trial.
            reduction_factor (int): Decicdes how many trials to promote to next rung.
                                    Only top 1 / reduction_factor of rung trials can be promoted.

        Raises:
            ValueError: If minimum resource is lower than maximum resource, raise an error.

        Returns:
            int: The number of rungs the bracket needs.
        """
        check_positive(minimum_resource, "minimum_resource")
        check_positive(maximum_resource, "maximum_resource")
        check_positive(reduction_factor, "reduction_factor")
        if minimum_resource > maximum_resource:
            raise ValueError(
                "maximum_resource should be bigger than minimum_resource. "
                f"but minimum_resource : {minimum_resource} / maximum_resource : {maximum_resource}"
            )

        return math.ceil(math.log(maximum_resource / minimum_resource, reduction_factor))

    def _initialize_rungs(self, hyper_parameter_configurations: List[AshaTrial]):
        num_trials = len(hyper_parameter_configurations)
        minimum_num_trials = self._reduction_factor**self.max_rung
        if minimum_num_trials > num_trials:
            raise ValueError(
                "number of hyper_parameter_configurations is not enough. "
                f"minimum number is {minimum_num_trials}, but current number is {num_trials}. "
                "if you want to let them be, you can decrease needed number "
                "by increasing reduction factor or minimum resource."
            )

        rungs = [
            Rung(
                self._minimum_resource * (self._reduction_factor**idx),
                math.floor(num_trials * (self._reduction_factor**-idx)),
                self._reduction_factor,
                idx,
            )
            for idx in range(self.max_rung + 1)
        ]

        for new_trial in hyper_parameter_configurations[: rungs[0].num_required_trial]:
            new_trial.bracket = self.id
            rungs[0].add_new_trial(new_trial)
            self._trials[new_trial.id] = new_trial

        return rungs

    def _promote_trial_if_available(self, rung_idx: int):
        check_not_negative(rung_idx, "rung_idx")

        if self.max_rung <= rung_idx:
            return None

        best_trial = self._rungs[rung_idx].get_trial_to_promote(self._asynchronous_sha, self._mode)
        if best_trial is not None:
            self._rungs[rung_idx + 1].add_new_trial(best_trial)

        return best_trial

    def get_next_trial(self) -> Optional[AshaTrial]:
        """Get next trial to train.

        Returns:
            Optional[AshaTrial]: Next trial to train. There is no trial to train, then return None.
        """
        current_rung = self.max_rung
        while current_rung >= 0:
            next_sample = self._promote_trial_if_available(current_rung)
            if next_sample is not None:
                if next_sample.is_done():
                    if current_rung < self.max_rung - 1:
                        current_rung += 1
                    continue
                break

            next_sample = self._rungs[current_rung].get_next_trial()
            if next_sample is not None:
                break

            current_rung -= 1

        return next_sample

    def is_done(self) -> bool:
        """Check that the bracket is done.

        Returns:
            bool: Whether bracket is done or not.
        """
        return self._rungs[-1].is_done()

    def get_best_trial(self) -> Optional[AshaTrial]:
        """Get best trial in the bracket.

        Returns:
            Optional[AshaTrial]: Best trial in the bracket. If there is no trial to select, then return None.
        """
        if not self.is_done():
            logger.warning("Bracket is not done yet.")

        trial = None
        for rung in reversed(self._rungs):
            trial = rung.get_best_trial(self._mode)
            if trial is None:
                continue
            break

        return trial

    def save_results(self, save_path: str):
        """Save a bracket result to 'save_path'.

        Args:
            save_path (str): Path where to save a bracket result.
        """
        result = self._get_result()
        with open(osp.join(save_path, "rung_status.json"), "w", encoding="utf-8") as f:
            json.dump(result, f)

        for trial_id, trial in self._trials.items():
            trial.save_results(osp.join(save_path, f"{trial_id}.json"))

    def print_result(self):
        """Print a bracket result."""
        print("*" * 20, f"{self.id} bracket", "*" * 20)
        result = self._get_result()
        del result["rung_status"]
        for key, val in result.items():
            print(f"{key} : {val}")

        best_trial = self.get_best_trial()
        if best_trial is None:
            print("This bracket isn't started yet!\n")
            return
        print(
            f"best trial:\n"
            f"id : {best_trial.id} / score : {best_trial.get_best_score()} / config : {best_trial.configuration}"
        )

        print("all trials:")
        for trial in self._trials.values():
            print(f"id : {trial.id} / score : {trial.get_best_score()} / config : {trial.configuration}")
        print()

    def _get_result(self):
        return {
            "minimum_resource": self._minimum_resource,
            "maximum_resource": self.maximum_resource,
            "reduction_factor": self._reduction_factor,
            "mode": self._mode,
            "asynchronous_sha": self._asynchronous_sha,
            "num_trials": len(self._trials),
            "rung_status": [
                {
                    "rung_idx": rung.rung_idx,
                    "num_trial": rung.get_num_trials(),
                    "num_required_trial": rung.num_required_trial,
                    "resource": rung.resource,
                }
                for rung in self._rungs
            ],
        }


class HyperBand(HpoBase):
    """It implements the Asyncronous HyperBand scheduler with iterations only.

    Please refer the below papers for the detailed algorithm.

    [1] "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization", JMLR 2018
        https://arxiv.org/abs/1603.06560
        https://homes.cs.washington.edu/~jamieson/hyperband.html

    [2] "A System for Massively Parallel Hyperparameter Tuning", MLSys 2020
        https://arxiv.org/abs/1810.05934

    Args:
        minimum_resource (Union[float, int]): Minimum resource to use for training a trial. Defaults to None.
        reduction_factor (int, optional): Decicdes how many trials to promote to next rung.
                                          Only top 1 / reduction_factor of rung trials can be promoted. Defaults to 3.
        asynchronous_sha (bool, optional): Whether to operate SHA asynchronously. Defaults to True.
        asynchronous_bracket (bool, optional): Whether SHAs(brackets) are running parallelly or not.
                                               Defaults to True. Defaults to False.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        minimum_resource: Optional[Union[int, float]] = None,
        reduction_factor: int = 3,
        asynchronous_sha: bool = True,
        asynchronous_bracket: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if minimum_resource is not None:
            check_positive(minimum_resource, "minimum_resource")
        _check_reduction_factor_value(reduction_factor)

        self._next_trial_id = 0
        self._reduction_factor = reduction_factor
        self._minimum_resource = minimum_resource
        self._asynchronous_sha = asynchronous_sha
        self._asynchronous_bracket = asynchronous_bracket
        self._trials: Dict[str, AshaTrial] = {}
        self._brackets: Dict[int, Bracket] = {}

        if not self._need_to_find_resource_value():
            self._brackets = self._make_brackets()

    def _need_to_find_resource_value(self) -> bool:
        return self.maximum_resource is None or self._minimum_resource is None

    def _make_brackets(self) -> Dict[int, Bracket]:
        if self.expected_time_ratio is None:
            brackets_config = self._make_default_brackets_setting()
        else:
            brackets_config = self.auto_config()
        return self._make_brackets_as_config(brackets_config)

    def _calculate_bracket_resource(self, num_max_rung_trials: int, bracket_index: int) -> Union[int, float]:
        """Calculate how much resource is needed for the bracket given that resume is available."""
        num_trial = self._calculate_num_bracket_trials(num_max_rung_trials, bracket_index)
        minimum_resource = self.maximum_resource * (self._reduction_factor**-bracket_index)

        total_resource = 0
        num_rungs = (
            Bracket.calcuate_max_rung_idx(
                minimum_resource, self.maximum_resource, self._reduction_factor  # type: ignore
            )
            + 1
        )
        previous_resource = 0
        resource = minimum_resource
        for _ in range(num_rungs):
            total_resource += num_trial * (resource - previous_resource)
            num_trial //= self._reduction_factor
            previous_resource = resource
            resource *= self._reduction_factor

        return total_resource

    def _calculate_num_bracket_trials(self, num_max_rung_trials: int, bracket_index: int) -> int:
        return num_max_rung_trials * (self._reduction_factor**bracket_index)

    def _calculate_origin_num_trial_for_bracket(self, bracket_idx: int) -> int:
        return self._calculate_num_bracket_trials(self._get_num_max_rung_trials(bracket_idx), bracket_idx)

    def _get_num_max_rung_trials(self, bracket_idx: int) -> int:
        return math.floor((self._calculate_s_max() + 1) / (bracket_idx + 1))

    def _calculate_s_max(self) -> int:
        return math.floor(
            math.log(self.maximum_resource / self._minimum_resource, self._reduction_factor)  # type: ignore
        )

    def _make_default_brackets_setting(self) -> List[Dict[str, Any]]:
        # Bracket order is the opposite of order of paper's.
        # This is for running default hyper parmeters with abundant resource.
        brackets_setting = []
        for idx in range(self._calculate_s_max() + 1):
            brackets_setting.append(
                {"bracket_index": idx, "num_trials": self._calculate_origin_num_trial_for_bracket(idx)}
            )

        return brackets_setting

    def _make_brackets_as_config(self, brackets_settings: List[Dict]) -> Dict[int, Bracket]:
        brackets = {}
        total_num_trials = 0
        for bracket_setting in brackets_settings:
            total_num_trials += bracket_setting["num_trials"]
        reserved_trials = list(self._trials.values()) if self._trials else []
        if len(reserved_trials) > total_num_trials:
            reserved_trials = reserved_trials[:total_num_trials]
        configurations = self._make_new_hyper_parameter_configs(total_num_trials - len(reserved_trials))

        for bracket_setting in brackets_settings:
            bracket_idx = bracket_setting["bracket_index"]
            num_trial_to_initialize = bracket_setting["num_trials"]
            minimum_resource = self.maximum_resource * (self._reduction_factor**-bracket_idx)

            bracket_configurations = []
            for reserved_trial in reserved_trials:
                if (
                    reserved_trial.bracket is None and reserved_trial.get_progress() <= minimum_resource
                ) or reserved_trial.bracket == bracket_idx:
                    num_trial_to_initialize -= 1
                    bracket_configurations.append(reserved_trial)
                    if len(bracket_configurations) >= bracket_setting["num_trials"]:
                        break

            for selected_trial in bracket_configurations:
                reserved_trials.remove(selected_trial)

            bracket_configurations.extend(configurations[:num_trial_to_initialize])
            configurations = configurations[num_trial_to_initialize:]

            bracket = Bracket(
                bracket_idx,
                minimum_resource,
                self.maximum_resource,  # type: ignore
                bracket_configurations,
                self._reduction_factor,
                self.mode,
                self._asynchronous_sha,
            )
            brackets[bracket_idx] = bracket

        return brackets

    def _make_new_hyper_parameter_configs(self, num: int) -> List[AshaTrial]:
        check_not_negative(num, "num")

        hp_configs: List[AshaTrial] = []
        if num == 0:
            return hp_configs

        hp_configs.extend(self._get_prior_hyper_parameters(num))
        if num - len(hp_configs) > 0:
            hp_configs.extend(self._get_random_hyper_parameter(num - len(hp_configs)))

        return hp_configs

    def _get_prior_hyper_parameters(self, num_samples: int) -> List[AshaTrial]:
        hp_configs = []
        num_samples = min([num_samples, len(self.prior_hyper_parameters)])
        for _ in range(num_samples):
            hyper_parameter = self.prior_hyper_parameters.pop(0)
            hp_configs.append(self._make_trial(hyper_parameter))

        return hp_configs

    def _get_random_hyper_parameter(self, num_samples: int) -> List[AshaTrial]:
        hp_configs = []
        latin_hypercube = LatinHypercube(len(self.search_space))
        configurations = latin_hypercube.random(num_samples)
        for config in configurations:
            config_with_key = {key: config[idx] for idx, key in enumerate(self.search_space)}
            hp_configs.append(
                self._make_trial(self.search_space.convert_from_zero_one_scale_to_real_space(config_with_key))
            )

        return hp_configs

    def _make_trial(self, hyper_parameter: Dict) -> AshaTrial:
        trial_id = self._get_new_trial_id()
        trial = AshaTrial(trial_id, hyper_parameter, self._get_train_environment())
        self._trials[trial_id] = trial
        return trial

    def _get_new_trial_id(self) -> str:
        trial_id = self._next_trial_id
        self._next_trial_id += 1
        return str(trial_id)

    def _get_train_environment(self) -> Dict:
        train_environment = {"subset_ratio": self.subset_ratio}
        return train_environment

    def get_next_sample(self) -> Optional[AshaTrial]:
        """Get next trial to train.

        Returns:
            Optional[AshaTrial]: Next trial to train. If there is no trial to train, then return None.
        """
        if not self._brackets:
            return self._make_trial_to_estimate_resource()

        next_sample = None
        for bracket in self._brackets.values():
            if not bracket.is_done():
                next_sample = bracket.get_next_trial()
                if self._asynchronous_bracket and next_sample is None:
                    continue
                break

        return next_sample

    def _make_trial_to_estimate_resource(self) -> AshaTrial:
        """Trial to estimate a maximum resource or minimum resource."""
        trial = self._make_new_hyper_parameter_configs(1)[0]
        if self.maximum_resource is None:
            if len(self._trials) == 1:  # first trial to estimate
                trial.bracket = 0
                trial.iteration = self.num_full_iterations
                trial.estimating_max_resource = True
            elif self._minimum_resource is not None:
                trial.iteration = self._minimum_resource
            else:
                trial.iteration = self.num_full_iterations
        else:
            trial.iteration = self.maximum_resource
        return trial

    def save_results(self):
        """Save a ASHA result."""
        for idx, bracket in self._brackets.items():
            save_path = osp.join(self.save_path, str(idx))
            os.makedirs(save_path, exist_ok=True)
            bracket.save_results(save_path)

    def auto_config(self) -> List[Dict[str, Any]]:
        """Configure ASHA automatically aligning with possible resource.

        Configure ASHA automatically. If resource is lesser than full ASHA, decrease ASHA scale.
        In contrast, resource is more than full ASHA, increase ASHA scale.

        Returns:
            List[Dict[str, Any]]: ASHA configuration. It's used to make brackets.
        """
        if self._trials:
            self._adjust_minimum_resource()
        if self._need_to_dcrease_hyerpband_scale():
            return self._decrease_hyperband_scale()
        return self._increase_hyperband_scale()

    def _adjust_minimum_resource(self):
        """Set meaningful minimum resource.

        Goal of this function is to avoid setting minimum resource too low
        to distinguish which trial is better.
        """
        if self.maximum_resource < self._reduction_factor:
            logger.debug("maximum_resource is lesser than reduction factor. adjusting minimum resource is skipped.")
            return

        trial = None
        for trial in self._trials.values():
            if trial.is_done():
                break
        if trial is None:
            logger.debug("There is no finished trial. adjusting minimum resource is skipped.")
            return

        cur_score = 0
        best_score = 0
        minimum_resource = 0
        for resource, score in trial.score.items():
            if resource > self.maximum_resource // self._reduction_factor:
                break
            cur_score = cur_score * 0.5 + score * 0.5
            if not left_vlaue_is_better(best_score, cur_score, self.mode):
                best_score = cur_score
                if minimum_resource == 0:
                    minimum_resource = resource
            else:
                minimum_resource = 0

        if minimum_resource == 0:
            minimum_resource = self.maximum_resource // self._reduction_factor
        self._minimum_resource = minimum_resource

    def _get_full_asha_resource(self) -> Union[int, float]:
        total_resource: Union[int, float] = 0
        for idx in range(self._calculate_s_max() + 1):
            num_max_rung_trials = self._get_num_max_rung_trials(idx)
            total_resource += self._calculate_bracket_resource(num_max_rung_trials, idx)

        return total_resource

    def _need_to_dcrease_hyerpband_scale(self) -> bool:
        """Check full ASHA resource exceeds expected_time_ratio."""
        if self.expected_time_ratio is None:
            return False

        return self._get_full_asha_resource() > self._get_expected_total_resource()

    def _decrease_hyperband_scale(self) -> List[Dict[str, Any]]:
        """Decrease Hyperband scale.

        From bracket which has biggest number of rung, check that it's resource exceeds expected_time_ratio
        if bracket is added. If not, bracket is added. If it does, check that number of trials for bracket
        can be reduced. if not, skip that bracket and check that next bracket can be added by same method.
        """
        brackets_setting: List = []
        total_resource: Union[int, float] = 0
        resource_upper_bound = self._get_expected_total_resource()

        reserved_resource: Union[int, float] = 0
        if self._trials:  # reserve resources for trials which should be run on bracket 0
            for trial in self._trials.values():
                if trial.bracket == 0:
                    reserved_resource += self.maximum_resource  # type: ignore
            total_resource += reserved_resource

        for idx in range(self._calculate_s_max(), -1, -1):
            if self._trials and idx == 0:
                total_resource -= reserved_resource

            origin_num_max_rung_trials = self._get_num_max_rung_trials(idx)
            for num_max_rung_trials in range(origin_num_max_rung_trials, 0, -1):
                bracket_resource = self._calculate_bracket_resource(num_max_rung_trials, idx)

                if total_resource + bracket_resource <= resource_upper_bound:
                    total_resource += bracket_resource
                    num_bracket_trials = self._calculate_num_bracket_trials(num_max_rung_trials, idx)
                    brackets_setting.insert(0, {"bracket_index": idx, "num_trials": num_bracket_trials})
                    break

        return brackets_setting

    def _get_expected_total_resource(self):
        if self.expected_time_ratio is None:
            raise ValueError("expected time ratio should be set to get expceted total resource")
        return (
            self.num_full_iterations
            * self.expected_time_ratio
            * self.acceptable_additional_time_ratio
            * self.num_workers
        )

    def _increase_hyperband_scale(self) -> List[Dict[str, Any]]:
        total_resource: Union[int, float] = 0
        bracket_status = {}
        s_max = self._calculate_s_max()

        # If all brackets can run more than one, then multiply number of trials on each bracket as many as possible
        sum_unit_resource: Union[int, float] = 0
        for idx in range(s_max + 1):
            num_max_rung_trials = self._get_num_max_rung_trials(idx)
            unit_resource = self._calculate_bracket_resource(1, idx)
            sum_unit_resource += unit_resource
            bracket_status[idx] = {"num_max_rung_trials": num_max_rung_trials, "unit_resource": unit_resource}
            total_resource += num_max_rung_trials * unit_resource

        maximum_reseource = self._get_expected_total_resource()
        available_num_trials = int((maximum_reseource - total_resource) // sum_unit_resource)

        for idx in range(s_max + 1):
            bracket_status[idx]["num_max_rung_trials"] += available_num_trials
        total_resource += sum_unit_resource * available_num_trials

        # add trials to brackets from big index as many as possible
        while True:
            update_flag = False
            for idx in range(s_max, -1, -1):
                if total_resource + bracket_status[idx]["unit_resource"] < maximum_reseource:
                    total_resource += bracket_status[idx]["unit_resource"]
                    bracket_status[idx]["num_max_rung_trials"] += 1
                    update_flag = True

            if not update_flag:
                break

        # set brackets setting
        brackets_setting = []
        for idx in range(s_max + 1):
            brackets_setting.append(
                {
                    "bracket_index": idx,
                    "num_trials": self._calculate_num_bracket_trials(
                        bracket_status[idx]["num_max_rung_trials"], idx  # type: ignore
                    ),
                }
            )

        return brackets_setting

    def _get_used_resource(self) -> Union[int, float]:
        used_resource: Union[int, float] = 0
        for trial in self._trials.values():
            used_resource += trial.get_progress()

        return used_resource

    def get_progress(self) -> Union[int, float]:
        """Get current progress of ASHA."""
        if self.is_done():
            return 1

        if self.expected_time_ratio is None:
            total_resource = self._get_full_asha_resource()
        else:
            total_resource = self._get_expected_total_resource()

        progress = self._get_used_resource() / total_resource

        return min(progress, 0.99)

    def report_score(
        self, score: Union[float, int], resource: Union[float, int], trial_id: str, done: bool = False
    ) -> Literal[TrialStatus.STOP, TrialStatus.RUNNING]:
        """Report a score to ASHA.

        Args:
            score (Union[float, int]): Score to report.
            resource (Union[float, int]): Resource used to get score.
            trial_id (str): Trial id.
            done (bool, optional): Whether training trial is done. Defaults to False.

        Returns:
            Literal[TrialStatus.STOP, TrialStatus.RUNNING]: Decide whether to continue training or not.
        """
        trial = self._trials[trial_id]
        if done:
            if self.maximum_resource is None and trial.estimating_max_resource:
                self.maximum_resource = trial.get_progress()
                self.num_full_iterations = self.maximum_resource
                if not self._need_to_find_resource_value():
                    self._brackets = self._make_brackets()
            trial.finalize()
        else:
            trial.register_score(score, resource)
            if self._minimum_resource is None:
                self._minimum_resource = min(trial.score.keys())
                if not self._need_to_find_resource_value():
                    self._brackets = self._make_brackets()
            if trial.is_done() or trial.bracket is None:
                return TrialStatus.STOP

        return TrialStatus.RUNNING

    def is_done(self) -> bool:
        """Check that the ASHA is done.

        Returns:
            bool: Whether ASHA is done.
        """
        if not self._brackets:
            return False
        for bracket in self._brackets.values():
            if not bracket.is_done():
                return False
        return True

    def get_best_config(self) -> Optional[Dict[str, Any]]:
        """Get best configuration in ASHA.

        Returns:
            Optional[Dict[str, Any]]: Best configuration in ASHA. If there is no trial to select, return None.
        """
        best_score = None
        best_trial = None

        for trial in self._trials.values():
            score = trial.get_best_score()
            if score is not None and (best_score is None or left_vlaue_is_better(score, best_score, self.mode)):
                best_score = score
                best_trial = trial

        if best_trial is None:
            return None
        return {"id": best_trial.id, "config": best_trial.configuration}

    def print_result(self):
        """Print a ASHA result."""
        print(
            "HPO(ASHA) result summary\n"
            f"Best config : {self.get_best_config()}.\n"
            f"Hyper band runs {len(self._brackets)} brackets.\n"
            "Brackets summary:"
        )
        for bracket in self._brackets.values():
            bracket.print_result()
