import json
import math
import os
from os import path as osp
from typing import Any, Dict, List, Optional, Union

from scipy.stats.qmc import LatinHypercube

from hpopt.hpo_base import HpoBase, Trial
from hpopt.hpo_base import TrialStatus
from hpopt.logger import get_logger
from hpopt.utils import (
    check_mode_input,
    check_not_negative,
    check_positive,
    left_is_better,
)

logger = get_logger()


def _check_reduction_factor_value(reduction_factor: int):
    if reduction_factor < 2:
        raise ValueError(
            "reduction_factor should be at least 2.\n"
            f"your value : {reduction_factor}"
            )

class AshaTrial(Trial):
    def __init__(
        self,
        id: Any,
        configuration: Dict,
        train_environment: Optional[Dict] = None
    ):
        super().__init__(id, configuration, train_environment)
        self._rung = None
        self._bracket = None

    @property
    def rung(self):
        return self._rung
        
    @rung.setter
    def rung(self, val: int):
        check_not_negative(val, "rung")
        self._rung = val

    @property
    def bracket(self):
        return self._bracket

    @bracket.setter
    def bracket(self, val: int):
        check_not_negative(val, "bracket")
        self._bracket = val

    def save_results(self, save_path: str):
        results = {
            "id" : self.id,
            "rung" : self.rung,
            "configuration" : self.configuration,
            "train_environment" : self.train_environment,
            "score" : {resource : score for resource, score in self.score.items()}
        }

        with open(save_path, "w") as f:
            json.dump(results, f)


class Rung:
    def __init__(
        self,
        resource: int,
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
        return self._num_required_trial

    @property
    def resource(self):
        return self._resource

    @property
    def rung_idx(self):
        return self._rung_idx

    def add_new_trial(self, trial: AshaTrial):
        if not self.need_more_trials():
            raise RuntimeError(f"{self.rung_idx} rung has already sufficient trials.")
        trial.iteration = self.resource
        trial.rung = self.rung_idx
        trial.status = TrialStatus.READY
        self._trials.append(trial)

    def get_best_trial(self, mode: str = "max"):
        check_mode_input(mode)
        best_score = None
        best_trial = None
        for trial in self._trials:
            if trial.rung != self.rung_idx:
                continue
            trial_score = trial.get_best_score(mode, self.resource)
            if (
                trial_score is not None
                and (best_score is None or left_is_better(trial_score, best_score, mode))
            ):
                best_trial = trial
                best_score = trial_score

        return best_trial

    def need_more_trials(self):
        return self.num_required_trial > self.get_num_trials()

    def get_num_trials(self):
        return len(self._trials)

    def is_done(self):
        if self.need_more_trials():
            return False
        for trial in self._trials:
            if not trial.is_done():
                return False
        return True

    def get_trial_to_promote(self, asynchronous_sha: bool = False, mode: str = "max"):
        num_finished_trial = 0
        num_promoted_trial = 0
        best_score = None
        best_trial = None

        for trial in self._trials:
            if trial.rung == self._rung_idx:
                if trial.is_done() and trial.status != TrialStatus.RUNNING:
                    num_finished_trial += 1
                    trial_score = trial.get_best_score(mode, self.resource)
                    if best_score is None or left_is_better(trial_score, best_score, mode):
                        best_trial = trial
                        best_score = trial_score
            else:
                num_promoted_trial += 1

        if asynchronous_sha:
            if (num_promoted_trial + num_finished_trial) // self._reduction_factor > num_promoted_trial:
                return best_trial
        else:
            if (
                self.is_done()
                and self._num_required_trial // self._reduction_factor > num_promoted_trial
            ):
                return best_trial

        return None

    def get_next_trial(self):
        for trial in self._trials:
            if not trial.is_done() and trial.status != TrialStatus.RUNNING:
                return trial

class Bracket:
    def __init__(
        self,
        id: int,
        minimum_resource: Union[float, int],
        maximum_resource: Union[float, int],
        hyper_parameter_configurations: List[AshaTrial],
        reduction_factor: int = 3,
        mode: str = "max",
        asynchronous_sha: bool = True
    ):
        check_positive(minimum_resource, "minimum_resource")
        _check_reduction_factor_value(reduction_factor)
        check_mode_input(mode)

        self._id = id
        self._minimum_resource = minimum_resource
        self.maximum_resource = maximum_resource
        self._reduction_factor = reduction_factor
        self._mode = mode
        self._asynchronous_sha = asynchronous_sha
        self._trials: Dict[int, AshaTrial] = {}
        self._rungs: List[Rung] = self._initialize_rungs(hyper_parameter_configurations)

    @property
    def id(self):
        return self._id

    @property
    def maximum_resource(self):
        return self._maximum_resource

    @maximum_resource.setter
    def maximum_resource(self, val: Union[float, int]):
        check_positive(val, "maximum_resource")
        if val < self._minimum_resource:
            raise ValueError(
                "maxnum_resource should be greater than minimum_resource.\n"
                f"value to set : {val}, minimum_resource : {self._minimum_resource}"
            )
        elif val == self._minimum_resource: 
            logger.warning("maximum_resource is same with the minimum_resource.")

        self._maximum_resource = val

    @property
    def max_rung(self):
        return self.calcuate_max_rung_idx(self._minimum_resource, self.maximum_resource, self._reduction_factor)

    @staticmethod
    def calcuate_max_rung_idx(
        minimum_resource: Union[float, int],
        maximum_resource: Union[float, int],
        reduction_factor: int
    ):
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
        minimum_num_trials = self._reduction_factor ** self.max_rung
        if minimum_num_trials > num_trials:
            raise ValueError(
                "number of hyper_parameter_configurations is not enough. "
                f"minimum number is {minimum_num_trials}, but current number is {num_trials}. "
                "if you want to let them be, you can decrease needed number "
                "by increasing reduction factor or minimum resource."
            )

        rungs = [
            Rung(
                self._minimum_resource * (self._reduction_factor ** idx),
                math.floor(num_trials * (self._reduction_factor ** -idx)),
                self._reduction_factor,
                idx,
            ) for idx in range(self.max_rung + 1)
        ]

        for new_trial in hyper_parameter_configurations[:rungs[0].num_required_trial]:
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
            self._rungs[rung_idx+1].add_new_trial(best_trial)

        return best_trial

    def get_next_trial(self):
        current_rung = self.max_rung
        while current_rung >= 0:
            next_sample = self._promote_trial_if_available(current_rung)
            if next_sample is not None:
                if next_sample.is_done():
                    if current_rung < self.max_rung - 1:
                        current_rung += 1
                    continue
                else:
                    break

            next_sample = self._rungs[current_rung].get_next_trial()
            if next_sample is not None:
                break

            current_rung -= 1

        return next_sample

    def is_done(self):
        return self._rungs[-1].is_done()

    def get_best_trial(self):
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
        result = self._get_result()
        with open(osp.join(save_path, "rung_status.json"), "w") as f:
            json.dump(result, f)

        for trial_id, trial in self._trials.items():
            trial.save_results(osp.join(save_path, f"{trial_id}.json"))

    def print_result(self):
        print("*"*20, f"{self.id} bracket", "*"*20)
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
            "minimum_resource" : self._minimum_resource,
            "maximum_resource" : self.maximum_resource,
            "reduction_factor" : self._reduction_factor,
            "mode" : self._mode,
            "asynchronous_sha" : self._asynchronous_sha,
            "num_trials" : len(self._trials),
            "rung_status" : [
                {
                    "rung_idx" : rung.rung_idx,
                    "num_trial" : rung.get_num_trials(),
                    "num_required_trial" : rung.num_required_trial,
                    "resource" : rung.resource

                } for rung in self._rungs
            ]
        }

class HyperBand(HpoBase):
    """
    This implements the Asyncronous HyperBand scheduler with iterations only.
    Please refer the below papers for the detailed algorithm.

    [1] "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization", JMLR 2018
        https://arxiv.org/abs/1603.06560
        https://homes.cs.washington.edu/~jamieson/hyperband.html

    [2] "A System for Massively Parallel Hyperparameter Tuning", MLSys 2020
        https://arxiv.org/abs/1810.05934

    Args:
        min_iterations (int): Only stop trials at least this old in time.
            The units are the same as the attribute named by `time_attr`.
        reduction_factor (float): Used to set halving rate and amount. This
            is simply a unit-less scalar.
    """

    def __init__(
        self,
        minimum_resource: Optional[Union[int, float]] = None,
        reduction_factor: int = 3,
        asynchronous_sha: bool = True,
        asynchronous_bracket: bool = False,
        **kwargs
    ):
        super(HyperBand, self).__init__(**kwargs)

        if minimum_resource is not None:
            check_positive(minimum_resource, "minimum_resource")
        _check_reduction_factor_value(reduction_factor)

        self._next_trial_id = 0
        self._reduction_factor = reduction_factor
        self._minimum_resource = minimum_resource
        self._asynchronous_sha = asynchronous_sha
        self._asynchronous_bracket = asynchronous_bracket
        self._trials: Dict[str, AshaTrial] = {}
        self._brackets: Dict[str, Bracket] = {}

        if not self._need_to_find_resource_value():
            self._brackets = self._make_brackets()

    def _need_to_find_resource_value(self):
        return self.maximum_resource is None or self._minimum_resource is None

    def _make_brackets(self):
        if self.expected_time_ratio is None:
            brackets_config = self._make_default_brackets_setting()
        else:
            brackets_config = self.auto_config()
        return self._make_brackets_as_config(brackets_config)

    def _calculate_bracket_resource(self, num_max_rung_trials: int, bracket_index: int):
        """calculate how much resource is needed for the bracket given that resume is available."""
        num_trial = self._calculate_num_bracket_trials(num_max_rung_trials, bracket_index)
        minimum_resource = self.maximum_resource * (self._reduction_factor ** -bracket_index)

        total_resource = 0
        num_rungs = Bracket.calcuate_max_rung_idx(minimum_resource, self.maximum_resource, self._reduction_factor) + 1
        previous_resource = 0
        resource = minimum_resource
        for _ in range(num_rungs):
            total_resource += num_trial * (resource - previous_resource)
            num_trial //= self._reduction_factor
            previous_resource = resource
            resource *= self._reduction_factor

        return total_resource

    def _calculate_num_bracket_trials(self, num_max_rung_trials: int, bracket_index: int):
        return num_max_rung_trials * (self._reduction_factor ** bracket_index)

    def _calculate_origin_num_trial_for_bracket(self, bracket_idx: int):
        return self._calculate_num_bracket_trials(self._get_num_max_rung_trials(bracket_idx), bracket_idx)

    def _get_num_max_rung_trials(self, bracket_idx: int):
        return math.floor((self._calculate_s_max() + 1) / (bracket_idx + 1))

    def _calculate_s_max(self):
        return math.floor(math.log(self.maximum_resource / self._minimum_resource, self._reduction_factor))

    def _make_default_brackets_setting(self):
        """
        bracket order is the opposite of order of paper's.
        this is for running default hyper parmeters with abundant resource.
        """
        brackets_setting = []
        for idx in range(self._calculate_s_max() + 1):
            brackets_setting.append(
                {"bracket_index" : idx, "num_trials":  self._calculate_origin_num_trial_for_bracket(idx)}
            )

        return brackets_setting

    def _make_brackets_as_config(self, brackets_settings: List[Dict]):
        brackets = {}
        total_num_trials = 0
        for bracket_setting in brackets_settings:
            total_num_trials += bracket_setting["num_trials"]
        reserved_trials = list(self._trials.values()) if self._trials else []
        if len(reserved_trials) >  total_num_trials:
            reserved_trials = reserved_trials[:total_num_trials]
        configurations = self._make_new_hyper_parameter_configs(total_num_trials - len(reserved_trials))

        for bracket_setting in brackets_settings:
            bracket_idx = bracket_setting["bracket_index"]
            num_trial_to_initialize = bracket_setting["num_trials"]
            minimum_resource = self.maximum_resource * (self._reduction_factor ** -bracket_idx)

            bracket_configurations = []
            for reserved_trial in reserved_trials:
                if (
                    (reserved_trial.bracket is None and reserved_trial.get_progress() <= minimum_resource)
                    or reserved_trial.bracket == bracket_idx 
                ):
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
                self.maximum_resource,
                bracket_configurations,
                self._reduction_factor,
                self.mode,
                self._asynchronous_sha
            )
            brackets[bracket_idx] = bracket

        return brackets

    def _make_new_hyper_parameter_configs(self, num: int):
        check_not_negative(num, "num")

        hp_configs = []
        if num == 0:
            return hp_configs

        if self.prior_hyper_parameters is not None:
            hp_configs.extend(self._get_prior_hyper_parameters(num))
        if num - len(hp_configs) > 0:
            hp_configs.extend(self._get_random_hyper_parameter(num-len(hp_configs)))

        return hp_configs

    def _get_prior_hyper_parameters(self, num_samples: int):
        hp_configs = []
        num_samples = min([num_samples, len(self.prior_hyper_parameters)])
        for _ in range(num_samples):
            hyper_parameter = self.prior_hyper_parameters.pop(0)
            hp_configs.append(self._make_trial(hyper_parameter))

        return hp_configs

    def _get_random_hyper_parameter(self, num_samples: int):
        hp_configs = []
        latin_hypercube = LatinHypercube(len(self.search_space))
        configurations = latin_hypercube.random(num_samples)
        for config in configurations:
            config_with_key = {key : config[idx] for idx, key in enumerate(self.search_space)}
            hp_configs.append(
                self._make_trial(self.search_space.convert_from_zero_one_scale_to_real_space(config_with_key))
            )

        return hp_configs

    def _make_trial(self, hyper_parameter: Dict):
        id = self._get_new_trial_id()
        trial = AshaTrial(id, hyper_parameter, self._get_train_environment())
        self._trials[id] = trial
        return trial

    def _get_new_trial_id(self):
        id = self._next_trial_id
        self._next_trial_id += 1
        return str(id)

    def _get_train_environment(self):
        train_environment = {"subset_ratio" : self.subset_ratio}
        return train_environment

    def get_next_sample(self):
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

    def _make_trial_to_estimate_resource(self):
        trial = self._make_new_hyper_parameter_configs(1)[0]
        if self.maximum_resource is None:
            if len(self._trials) == 1:  # first trial to estimate
                trial.bracket = 0
                trial.iteration = self.num_full_iterations
            elif self._minimum_resource is not None:
                trial.iteration = self._minimum_resource
            else:
                trial.iteration = self.num_full_iterations
        else:
            trial.iteration = self.maximum_resource
        return trial

    def save_results(self):
        for idx, bracket in self._brackets.items():
            save_path = osp.join(self.save_path, str(idx))
            os.makedirs(save_path, exist_ok=True)
            bracket.save_results(save_path)

    def auto_config(self):
        if self._trials:
            self._adjust_minimum_resource()
        if self._need_to_dcrease_hyerpband_scale():
            return self._decrease_hyperband_scale()
        else:
            return self._increase_hyperband_scale()

    def _adjust_minimum_resource(self):
        """
        Set meaningful minimum resource. Goal of this function is to avoid setting minimum resource too low
        to distinguish which trial is better.
        """
        if self.maximum_resource < self._reduction_factor:
            logger.debug("maximum_resource is less than reduction factor. adjusting minimum resource is skipped.")
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
            if not left_is_better(best_score, cur_score, self.mode):
                best_score = cur_score
                if minimum_resource == 0:
                    minimum_resource = resource
            else:
                minimum_resource = 0

        if minimum_resource == 0:
            minimum_resource = self.maximum_resource // self._reduction_factor
        self._minimum_resource = minimum_resource

    def _need_to_dcrease_hyerpband_scale(self):
        """check full ASHA resource exceeds expected_time_ratio."""
        if self.expected_time_ratio is None:
            return False

        total_resource = 0
        for idx in range(self._calculate_s_max()+1):
            num_max_rung_trials = self._get_num_max_rung_trials(idx)
            total_resource += self._calculate_bracket_resource(num_max_rung_trials, idx)

        return total_resource > self._get_expected_total_resource()

    def _decrease_hyperband_scale(self):
        """
        from bracket which has biggest number of rung, check that it's resource exceeds expected_time_ratio
        if bracket is added. If not, bracket is added. If it does, check that number of trials for bracket
        can be reduced. if not, skip that bracket and check that next bracket can be added by same method.
        """
        brackets_setting = []
        total_resource = 0
        resource_upper_bound = self._get_expected_total_resource()

        reserved_resource = 0
        if self._trials:  # reserve resources for trials which should be run on bracket 0
            for trial in self._trials.values():
                if trial.bracket == 0:
                    reserved_resource += self.maximum_resource
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
                    brackets_setting.insert(0, {"bracket_index" : idx, "num_trials" : num_bracket_trials})
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

    def _increase_hyperband_scale(self):
        total_resource = 0
        bracket_status = {}
        s_max = self._calculate_s_max()

        # If all brackets can run more than one, then multiply number of trials on each bracket as many as possible
        sum_unit_resource = 0
        for idx in range(s_max+1):
            num_max_rung_trials = self._get_num_max_rung_trials(idx)
            unit_resource = self._calculate_bracket_resource(1, idx)
            sum_unit_resource += unit_resource
            bracket_status[idx] = {"num_max_rung_trials" : num_max_rung_trials, "unit_resource" : unit_resource}
            total_resource += num_max_rung_trials * unit_resource

        maximum_reseource = self._get_expected_total_resource()
        available_num_trials = int((maximum_reseource - total_resource) // sum_unit_resource)

        for idx in range(s_max+1):
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
        for idx in range(s_max+1):
            brackets_setting.append(
                {
                    "bracket_index" : idx,
                    "num_trials" : self._calculate_num_bracket_trials(bracket_status[idx]["num_max_rung_trials"], idx)
                } 
            )

        return brackets_setting

    def get_progress(self):
        raise NotImplementedError

    def report_score(self, score: Union[float, int], resource: Union[float, int], trial_id: str, done: bool = False):
        trial = self._trials[trial_id]
        if done:
            if self.maximum_resource is None:
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

    def is_done(self):
        if not self._brackets:
            return False
        for bracket in self._brackets.values():
            if not bracket.is_done():
                return False
        return True

    def get_best_config(self):
        best_score = None
        best_trial = None

        for trial in self._trials.values():
            score = trial.get_best_score()
            if score is not None and (best_score is None or left_is_better(score, best_score, self.mode)):
                best_score = score
                best_trial = trial

        if best_trial is None:
            return None
        return {"id" : best_trial.id , "config" : best_trial.configuration}

    def print_result(self):
        print(
            "HPO(ASHA) result summary\n"
            f"Best config : {self.get_best_config()}.\n"
            f"Hyper band runs {len(self._brackets)} brackets.\n"
            "Brackets summary:"
        )
        for bracket in self._brackets.values():
            bracket.print_result()
