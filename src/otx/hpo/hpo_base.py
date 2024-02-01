# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""HPO algorithm abstract class."""

from __future__ import annotations

import json
import tempfile
import logging
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Any, Literal

from otx.hpo.search_space import SearchSpace
from otx.hpo.utils import check_mode_input, check_positive

logger = logging.getLogger(__name__)


class HpoBase(ABC):
    """Abstract class for HPO algorithms.

    This implements class for HPO algorithm base.
    Only common methods are implemented but not core algorithm of HPO.

    Args:
        search_space (dict[str, dict[str, Any]]): hyper parameter search space to find.
        save_path (str | None, optional): path where result of HPO is saved.
        mode ("max" | "min", optional): One of {min, max}. Determines whether objective is
                                        minimizing or maximizing the metric attribute.
        num_trials (int | None, optional): How many training to conduct for HPO.
        num_workers (int, optional): How many trains are executed in parallel.
        num_full_iterations (int, optional): epoch for traninig after HPO.
        non_pure_train_ratio (float, optional): ratio of validation time to (train time + validation time)
        full_dataset_size (int, optional): train dataset size
        metric (str, optional): Which score metric to use.
        expected_time_ratio (int | float | None, optional): Time to use for HPO.
                                                            If HPO is configured automatically,
                                                            HPO use time about exepected_time_ratio *
                                                            train time after HPO times.
        maximum_resource (int | float | None, optional): Maximum resource to use for training each trial.
        subset_ratio (float | int | None, optional): ratio to how many train dataset to use for each trial.
                                     The lower value is, the faster the speed is.
                                     But If it's too low, HPO can be unstable.
        min_subset_size (int, optional) : Minimum size of subset. Default value is 500.
        resume (bool, optional): resume flag decide to use previous HPO results.
                                 If HPO completed, you can just use optimized hyper parameters.
                                 If HPO stopped in middle, you can resume in middle.
        prior_hyper_parameters (dict | list[dict] | None, optional) = Hyper parameters to try first.
        acceptable_additional_time_ratio (float | int, optional) = Decide how much additional time can be acceptable.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        search_space: dict[str, dict[str, Any]],
        save_path: str | None = None,
        mode: Literal["max", "min"] = "max",
        num_trials: int | None = None,
        num_workers: int = 1,
        num_full_iterations: int | float = 1,
        non_pure_train_ratio: float = 0.2,
        full_dataset_size: int = 0,
        metric: str = "mAP",
        expected_time_ratio: int | float | None = None,
        maximum_resource: int | float | None = None,
        subset_ratio: float | int | None = None,
        min_subset_size: int = 500,
        resume: bool = False,
        prior_hyper_parameters: dict | list[dict] | None = None,
        acceptable_additional_time_ratio: float | int = 1.0,
    ) -> None:
        # pylint: disable=too-many-arguments, too-many-locals
        check_mode_input(mode)
        check_positive(full_dataset_size, "full_dataset_size")
        check_positive(num_full_iterations, "num_full_iterations")
        if not 0 < non_pure_train_ratio <= 1:
            raise ValueError(
                "non_pure_train_ratio should be greater than 0 and lesser than or equal to 1."
                f" Your value is {subset_ratio}"
            )
        if maximum_resource is not None:
            check_positive(maximum_resource, "maximum_resource")
        if num_trials is not None:
            check_positive(num_trials, "num_trials")
        check_positive(num_workers, "num_workers")
        if subset_ratio is not None:
            if not 0 < subset_ratio <= 1:
                raise ValueError(
                    "subset_ratio should be greater than 0 and lesser than or equal to 1."
                    f" Your value is {subset_ratio}"
                )

        if save_path is None:
            save_path = tempfile.mkdtemp(prefix="OTX-hpo-")
        self.save_path = save_path
        self.search_space = SearchSpace(search_space)
        self.mode = mode
        self.num_trials = num_trials
        self.num_workers = num_workers
        self.num_full_iterations = num_full_iterations
        self.non_pure_train_ratio = non_pure_train_ratio
        self.full_dataset_size = full_dataset_size
        self.expected_time_ratio = expected_time_ratio
        self.maximum_resource: int | float | None = maximum_resource
        self.subset_ratio = subset_ratio
        self.min_subset_size = min_subset_size
        self.resume = resume
        self.hpo_status: dict = {}
        self.metric = metric
        self.acceptable_additional_time_ratio = acceptable_additional_time_ratio
        if prior_hyper_parameters is None:
            prior_hyper_parameters = []
        elif isinstance(prior_hyper_parameters, dict):
            prior_hyper_parameters = [prior_hyper_parameters]
        self.prior_hyper_parameters = prior_hyper_parameters

    @abstractmethod
    def print_result(self):
        """Print a HPO algorithm result."""
        raise NotImplementedError

    @abstractmethod
    def save_results(self):
        """Save a HPO algorithm result."""
        raise NotImplementedError

    @abstractmethod
    def is_done(self):
        """Check whether HPO algorithm is done."""
        raise NotImplementedError

    @abstractmethod
    def get_next_sample(self):
        """Get next sample to train."""
        raise NotImplementedError

    @abstractmethod
    def auto_config(self):
        """Configure HPO algorithm automatically."""
        raise NotImplementedError

    @abstractmethod
    def get_progress(self):
        """Get current progress of HPO algorithm."""
        raise NotImplementedError

    @abstractmethod
    def report_score(self, score, resource, trial_id, done):
        """Report a score to HPO algorithm."""
        raise NotImplementedError

    @abstractmethod
    def get_best_config(self):
        """Get best config of HPO algorithm."""
        raise NotImplementedError


class Trial:
    """Trial to train with given hyper parameters.

    Args:
        trial_id (Any): Trial id.
        configuration (dict): Configuration to train with.
        train_environment (dict | None, optional): Train environment for the trial. Defaults to None.
    """

    def __init__(self, trial_id: Any, configuration: dict, train_environment: dict | None = None) -> None:
        self._id = trial_id
        self._configuration = configuration
        self.score: dict[float | int, float | int] = {}
        self._train_environment = train_environment
        self._iteration = None
        self.status: TrialStatus = TrialStatus.READY
        self._done = False

    @property
    def id(self) -> Any:
        """Trial id."""
        return self._id

    @property
    def configuration(self) -> dict:
        """Configuration to train with."""
        return self._configuration

    @property
    def iteration(self) -> int | float | None:
        """Iteration to use for training."""
        return self._iteration

    @iteration.setter
    def iteration(self, val: int | float) -> None:
        """Setter for iteration."""
        check_positive(val, "iteration")
        self._iteration = val
        if self.get_progress() < val:
            self._done = False

    @property
    def train_environment(self) -> dict | None:
        """Train environment for the trial."""
        return self._train_environment

    def get_train_configuration(self) -> dict[str, Any]:
        """Get configurations needed to trian."""
        self._configuration["iterations"] = self.iteration
        return {"id": self.id, "configuration": self.configuration, "train_environment": self.train_environment}

    def register_score(self, score: int | float, resource: int | float) -> None:
        """Register score to the trial.

        Args:
            score (int | float): Score to register.
            resource (int | float): Resource used to get score. It should be positive.
        """
        check_positive(resource, "resource")
        self.score[resource] = score

    def get_best_score(
        self, mode: Literal["max", "min"] = "max", resource_limit: float | int | None = None
    ) -> float | int | None:
        """Get best score of the trial.

        Args:
            mode ("max" | "min", optional):
                Decide which is better between highest score or lowest score. Defaults to "max".
            resource_limit (float | int | None, optional): Find a best score among the score at resource
                                                           lower than this value. Defaults to None.

        Returns:
            float | int | None: Best score. If there is no score, return None.
        """
        check_mode_input(mode)

        if resource_limit is None:
            scores = self.score.values()
        else:
            scores = [val for key, val in self.score.items() if key <= resource_limit]  # type: ignore

        if len(scores) == 0:
            return None

        if mode == "max":
            return max(scores)
        return min(scores)

    def get_progress(self) -> float | int:
        """Get a progress of the trial.

        Returns:
            float | int: How many resource is used for the trial.
        """
        if len(self.score) == 0:
            return 0
        return max(self.score.keys())

    def save_results(self, save_path: str) -> None:
        """Save a result in the 'save_path'.

        Args:
            save_path (str): Path where to save a result.
        """
        results = {
            "id": self.id,
            "configuration": self.configuration,
            "train_environment": self.train_environment,
            "score": self.score,
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f)

    def finalize(self) -> None:
        """Set done as True."""
        if not self.score:
            raise RuntimeError(f"Trial{self.id} didn't report any score but tries to be done.")
        self._done = True

    def is_done(self) -> bool:
        """Check the trial is done."""
        if self.iteration is None:
            raise ValueError("iteration isn't set yet.")
        return self._done or self.get_progress() >= self.iteration


class TrialStatus(IntEnum):
    """Enum class for trial status."""

    UNKNOWN = -1
    READY = 0
    RUNNING = 1
    STOP = 2
    CUDAOOM = 3
