# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import json
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Any, Dict, List, Optional, Union

from hpopt.logger import get_logger
from hpopt.search_space import SearchSpace
from hpopt.utils import check_mode_input, check_positive

logger = get_logger()


class HpoBase(ABC):
    """
    This implements class which make frame for bayesian optimization
    or ahsa class. So, only common methods are implemented but
    core method for HPO.

    Args:
        save_path (str): path where result of HPO is saved.
        search_space (list): hyper parameter search space to find.
        mode (str): One of {min, max}. Determines whether objective is
                    minimizing or maximizing the metric attribute.
        num_init_trials (int): Only for SMBO. How many trials to use to init SMBO.
        num_trials (int): How many training to conduct for HPO.
        num_workers (int): How many trains are executed in parallel.
        num_full_iterations (int): epoch for traninig after HPO.
        non_pure_train_ratio (float): ratio of validation time to (train time + validation time)
        full_dataset_size (int): train dataset size
        expected_time_ratio (int or float): Time to use for HPO.
                                            If HPO is configured automatically,
                                            HPO use time about exepected_time_ratio *
                                            train time after HPO times.
        max_iterations (int): Max training epoch for each trial.
        subset_ratio (float or int): ratio to how many train dataset to use for each trial.
                                     The lower value is, the faster the speed is.
                                     But If it's too low, HPO can be unstable.
        min_subset_size (float or int) : Minimum size of subset. Default value is 500.
        verbose (int): Decide how much content to print.
        resume (bool): resume flag decide to use previous HPO results.
                       If HPO completed, you can just use optimized hyper parameters.
                       If HPO stopped in middle, you can resume in middle.
    """

    def __init__(
        self,
        search_space: Dict[str, Dict[str, Any]],
        save_path: str = "/tmp/hpopt",
        mode: str = "max",
        num_trials: Optional[int] = None,
        num_workers: int = 1,
        num_full_iterations: int = 1,
        non_pure_train_ratio: float = 0.2,
        full_dataset_size: int = 0,
        metric: str = "mAP",
        expected_time_ratio: Optional[Union[int, float]] = None,
        maximum_resource: Optional[Union[int, float]] = None,
        subset_ratio: Optional[Union[float, int]] = None,
        min_subset_size=500,
        batch_size_name: Optional[str] = None,
        verbose: int = 0,
        resume: bool = False,
        prior_hyper_parameters: Optional[Union[Dict, List[Dict]]] = None,
        acceptable_additional_time_ratio: Union[float, int] = 1.0
    ):
        check_mode_input(mode)
        check_positive(full_dataset_size, "full_dataset_size")
        check_positive(num_full_iterations, "num_full_iterations")
        if not (0 < non_pure_train_ratio <= 1):
            raise ValueError(
                "non_pure_train_ratio should be between 0 and 1."
                f" Your value is {non_pure_train_ratio}"
            )
        if maximum_resource is not None:
            check_positive(maximum_resource, "maximum_resource")
        if num_trials is not None:
            check_positive(num_trials, "num_trials")
        check_positive(num_workers, "num_workers")
        if subset_ratio is not None:
            if not (0 < subset_ratio <= 1.0):
                raise ValueError(
                    "subset_ratio should be greater than 0 and lesser than or equal to 1."
                    f" Your value is {subset_ratio}"
                )

        self.save_path = save_path
        self.search_space = SearchSpace(search_space)
        self.mode = mode
        self.num_trials = num_trials
        self.num_workers = num_workers
        self.num_full_iterations = num_full_iterations
        self.non_pure_train_ratio = non_pure_train_ratio
        self.full_dataset_size = full_dataset_size
        self.expected_time_ratio = expected_time_ratio
        self.maximum_resource = maximum_resource
        self.subset_ratio = subset_ratio
        self.min_subset_size = min_subset_size
        self.verbose = verbose
        self.resume = resume
        self.hpo_status: dict = {}
        self.metric = metric
        self.batch_size_name = batch_size_name
        self.acceptable_additional_time_ratio = acceptable_additional_time_ratio
        self.prior_hyper_parameters = prior_hyper_parameters
        if isinstance(self.prior_hyper_parameters, dict):
            self.prior_hyper_parameters = [self.prior_hyper_parameters]

    def print_results(self):
        field_widths = []
        field_param_name = []
        print(f'|{"#": ^5}|', end="")
        for param in self.search_space:
            field_title = f"{param}"
            filed_width = max(len(field_title) + 2, 20)
            field_widths.append(filed_width)
            field_param_name.append(param)
            print(f"{field_title: ^{filed_width}} |", end="")
        print(f'{"score": ^21}|')

        for trial_id, config_item in enumerate(self.hpo_status["config_list"], start=1):
            if config_item["score"] is not None:
                print(f"|{trial_id: >4} |", end="")
                real_config = config_item["config"]
                for param, field_width in zip(field_param_name, field_widths):
                    print(f"{real_config[param]: >{field_width}} |", end="")
                score = config_item["score"]
                print(f"{score: >20} |", end="")
                print("")

    @abstractmethod
    def save_results(self):
        raise NotImplementedError

    @abstractmethod
    def is_done(self):
        raise NotImplementedError

    @abstractmethod
    def get_next_sample(self):
        raise NotImplementedError

    @abstractmethod
    def auto_config(self):
        raise NotImplementedError

    @abstractmethod
    def get_progress(self):
        raise NotImplementedError

    @abstractmethod
    def report_score(self, score, resource, trial_id, done):
        raise NotImplementedError

    @abstractmethod
    def get_best_config(self):
        raise NotImplementedError

    @abstractmethod
    def print_result(self):
        raise NotImplementedError

class Trial:
    def __init__(
        self,
        id: Any,
        configuration: Dict,
        train_environment: Optional[Dict] = None
    ):
        self._id = id
        self._configuration = configuration
        self.score: Dict[Union[float, int], Union[float, int]] = {}
        self._train_environment = train_environment
        self._iteration = None
        self.status: TrialStatus = TrialStatus.READY

    @property
    def id(self):
        return self._id

    @property
    def configuration(self):
        return self._configuration

    @property
    def iteration(self):
        return self._iteration

    @iteration.setter
    def iteration(self, val):
        check_positive(val, "iteration")
        self._iteration = val

    @property
    def train_environment(self):
        return self._train_environment

    def get_train_configuration(self):
        self._configuration["iterations"] = self.iteration
        return {
            "id" : self.id,
            "configuration" : self.configuration,
            "train_environment" : self.train_environment
        }

    def register_score(self, score: Union[int, float], resource: Union[int, float]):
        check_positive(resource, "resource")
        self.score[resource] = score

    def get_best_score(self, mode: str = "max", resource_limit: Optional[Union[float, int]] = None):
        check_mode_input(mode)

        if resource_limit is None:
            scores = self.score.values()
        else:
            scores = [val for key, val in self.score.items() if key <= resource_limit]

        if len(scores) == 0:
            return None

        if mode == "max":
            return max(scores)
        else:
            return min(scores)

    def get_progress(self):
        if len(self.score) == 0:
            return 0
        return max(self.score.keys())

    def save_results(self, save_path: str):
        results = {
            "id" : self.id,
            "configuration" : self.configuration,
            "train_environment" : self.train_environment,
            "score" : {resource : score for resource, score in self.score.items()}
        }

        with open(save_path, "w") as f:
            json.dump(results, f)

    def finalize(self):
        if self.get_progress() < self.iteration:
            best_score = self.get_best_score()
            if best_score is None:
                raise RuntimeError(f"Although {self.id} trial doesn't report any score but it's done")
            self.register_score(best_score, self.iteration)

    def is_done(self):
        if self.iteration is None:
            raise ValueError("iteration isn't set yet.")
        return self.get_progress() >= self.iteration

class TrialStatus(IntEnum):
    UNKNOWN = -1
    READY = 0
    RUNNING = 1
    STOP = 2
    CUDAOOM = 3
