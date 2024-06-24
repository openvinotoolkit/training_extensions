# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Search space class for HPO."""

from __future__ import annotations

import logging
import math
import typing
from typing import Any, Generator, Literal

from otx.hpo.utils import check_positive

logger = logging.getLogger()

AVAILABLE_SEARCH_SPACE_TYPE = [
    "uniform",
    "quniform",
    "loguniform",
    "qloguniform",
    "choice",
]


class SingleSearchSpace:
    """The class which implements single search space used for HPO.

    This class supports uniform and quantized uniform with normal and log scale
    in addition to categorical type. Quantized type has step which is unit for change.

    Args:
        type ("uniform" | "loguniform" | "quniform" | "qloguniform" | "choice"): type of hyper parameter.
        min (float | int | None, optional): upper bounding of search space.
                                            If type isn't choice, this value is required.
        max (float | int | None, optional): lower bounding of search space
                                            If type isn't choice, this value is required.
        step (int | None, optional): unit for change. If type is quniform or qloguniform,
                                     This value is required.
        log_base (int | None, optional): base of logarithm. Default value is 2.
        choice_list (list | tuple | None, optional): candidiates for choice type. If task is choice,
                                    this value is required.
    """

    def __init__(
        self,
        type: Literal["uniform", "loguniform", "quniform", "qloguniform", "choice"],  # noqa: A002
        min: float | int | None = None,  # noqa: A002
        max: float | int | None = None,  # noqa: A002
        step: float | int | None = None,
        log_base: int | None = 2,
        choice_list: list | tuple | None = None,
    ) -> None:
        # pylint: disable=redefined-builtin
        self._type = type
        if self.is_categorical():
            self._choice_list = choice_list
            self._align_min_max_to_choice_list_if_categorical()
        else:
            self._min = min
            self._max = max
            self._step = step
            self._log_base = log_base
        self._check_all_value_is_right()

    @property
    def type(  # noqa: A003
        self,
    ) -> Literal["uniform", "loguniform", "quniform", "qloguniform", "choice"]:
        """Type of hyper parameter in search space."""
        return self._type

    @property
    def min(self) -> float | int | None:  # noqa: A003
        """Lower bounding of search space."""
        return self._min

    @property
    def max(self) -> float | int | None:  # noqa: A003
        """Upper bounding of search space."""
        return self._max

    @property
    def step(self) -> float | int | None:
        """Unit for change."""
        return self._step

    @property
    def log_base(self) -> int | None:
        """Base of logarithm."""
        return self._log_base

    @property
    def choice_list(self) -> list | tuple | None:
        """Candidiates for choice type."""
        return self._choice_list

    def set_value(
        self,
        type: Literal["uniform", "loguniform", "quniform", "qloguniform", "choice"] | None = None,  # noqa: A002
        min: float | int | None = None,  # noqa: A002
        max: float | int | None = None,  # noqa: A002
        step: float | int | None = None,
        log_base: int | None = None,
        choice_list: list | tuple | None = None,
    ) -> None:
        # pylint: disable=redefined-builtin
        """Set attributes of the class.

        There are some dependencies when setting attributes of this class.
        So, modifying a single attribute at a time can set wrong value.
        For example, `max` can be set as a value lower than `min`.
        To prevent it, this function priovides a way to set all necessary values at a time.

        Args:
            type ("uniform" | "loguniform" | "quniform" | "qloguniform" | "choice" | None, optional):
                type of hyper parameter in search space. supported types: uniform, loguniform, quniform,
                qloguniform, choice
            min (float | int | None, optional): upper bounding of search space.
                                                If type isn't choice, this value is required.
            max (float | int | None, optional): lower bounding of search space
                                                If type isn't choice, this value is required.
            step (int | None, optional): unit for change. If type is quniform or qloguniform,
                                         This value is required.
            log_base (int | None, optional): base of logarithm. Default value is 2.
            choice_list (list | tuple, optional): candidiates for choice type. If task is choice,
                                                  this value is required.
        """
        if type is not None:
            self._type = type
        if min is not None:
            self._min = min
        if max is not None:
            self._max = max
        if step is not None:
            self._step = step
        if log_base is not None:
            self._log_base = log_base
        if choice_list is not None:
            self._choice_list = choice_list
        if self.is_categorical():
            self._align_min_max_to_choice_list_if_categorical()

        self._check_all_value_is_right()

    def _align_min_max_to_choice_list_if_categorical(self) -> None:
        if self.is_categorical():
            self._min = 0
            self._max = len(self._choice_list) - 1  # type: ignore[arg-type]

    def _check_all_value_is_right(self) -> None:
        # pylint: disable=too-many-branches
        if self._type not in AVAILABLE_SEARCH_SPACE_TYPE:
            error_msg = (
                f"type should be one of {', '.join(AVAILABLE_SEARCH_SPACE_TYPE)}. But your argument is {self._type}"
            )
            raise ValueError(error_msg)

        if self.is_categorical():
            if self._choice_list is None or len(self._choice_list) <= 1:
                error_msg = "If type is choice, choice_list should have more than one element"
                raise ValueError(error_msg)
            if self._min != 0:
                error_msg = "if type is categorical, min should be 0."
                raise ValueError(error_msg)
            if self._max != len(self._choice_list) - 1:
                error_msg = "if type is categorical, max should be last index number of choice_list."
                raise ValueError(error_msg)
        else:
            if self._min is None:
                error_msg = "If type isn't choice, you should set min value of search space."
                raise ValueError(error_msg)
            if self._max is None:
                error_msg = "If type isn't choice, you should set max value of search space."
                raise ValueError(error_msg)

            if self._min >= self._max:
                error_msg = (
                    f"max value should be greater than min value.\nmax value : {self._max} / min value : {self._min}"
                )
                raise ValueError(error_msg)

            if self.use_log_scale():
                if self._log_base is None:
                    error_msg = "Type loguniform and qloguniform need log_base."
                    raise ValueError(error_msg)
                if self._log_base <= 1:
                    error_msg = f"log base should be greater than 1.\nyour log base value is {self._log_base}."
                    raise ValueError(error_msg)
                if self._min <= 0:
                    error_msg = (
                        f"If you use log scale, min value should be greater than 0.\nyour min value is {self._min}"
                    )
                    raise ValueError(error_msg)
            if self.use_quantized_step():
                if self._step is None:
                    error_msg = f"The {self._type} type requires step value. But it doesn't exists"
                    raise ValueError(error_msg)
                check_positive(self._step, "step")
                if self._step > self._max - self._min:
                    error_msg = (
                        "Difference between min and max is greater than step.\n"
                        f"Current value is min : {self._min}, max : {self._max}, step : {self._step}"
                    )
                    raise ValueError(error_msg)

    def __repr__(self) -> str:
        """Print serach space status."""
        if self.is_categorical():
            return f"type: {self._type}, candidiate : {', '.join(self._choice_list)}"  # type: ignore[arg-type]
        rep = f"type: {self._type}, search space : {self._min} ~ {self._max}"
        if self.use_quantized_step():
            rep += f", step : {self._step}"
        if self.use_log_scale():
            rep += f", log base : {self._log_base}"
        return rep

    def is_categorical(self) -> bool:
        """Check current instance is categorical type."""
        return self._type == "choice"

    def use_quantized_step(self) -> bool:
        """Check current instance is one of type to use `step`."""
        return self._type in ("quniform", "qloguniform")

    def use_log_scale(self) -> bool:
        """Check current instance is one of type to use `log scale`."""
        return self._type in ("loguniform", "qloguniform")

    def lower_space(self) -> float | int | None:
        """Get lower bound value considering log scale if necessary."""
        if self.use_log_scale():
            return math.log(self._min, self._log_base)  # type: ignore[arg-type]
        return self._min

    def upper_space(self) -> float | int | None:
        """Get upper bound value considering log scale if necessary."""
        if self.use_log_scale():
            return math.log(self._max, self._log_base)  # type: ignore[arg-type]
        return self._max

    @typing.no_type_check
    def space_to_real(self, number: int | float) -> int | float:
        """Convert search space from HPO perspective to human perspective.

        Args:
            number (Union[int, float]): Value to convert

        Returns:
            Union[int, float]: value converted to human perspective.
        """
        if self.is_categorical():
            idx = max(min(int(number), len(self._choice_list) - 1), 0)
            return self._choice_list[idx]
        if self.use_log_scale():
            number = self._log_base**number
        if self.use_quantized_step():
            gap = self._min % self._step
            number = round((number - gap) / self._step) * self._step + gap
        return number

    def real_to_space(self, number: int | float) -> int | float:
        """Convert search space from human perspective to HPO perspective.

        Args:
            number (Union[int, float]): Value to convert

        Returns:
            Union[int, float]: value converted to HPO perspective.
        """
        if self.use_log_scale():
            return math.log(number, self._log_base)  # type: ignore[arg-type]
        return number


class SearchSpace:
    """Class which manages all search spaces of hyper parameters to optimize.

    This class supports HPO algorithms by providing necessary functionalities.

    Args:
        search_space (dict) :
            search spaces of hyper parameters to optimize.
            arguemnt format is as bellow.
            {
                "some_hyper_parameter_name" : {
                    "type": type of search space of hyper parameter.
                                  supported types: uniform, loguniform, quniform,
                                  qloguniform or choice
                    # At this point, there are two available formats.
                    # first format is adding each key, value pairs depending on type as bellow.
                    "max" : upper_bound_value,
                    "min" : lower_bound_value,
                    ...
                    # available keys are max, min, step, log_base, choice_list.
                    # second format is adding a list whose key is "range" which contains
                    # necessary values mentioned above with proper order as bellow.
                    "range" (list): range of hyper parameter search space.
                                    What value at each position means is as bellow.
                                    uniform: [min, max]
                                    quniform: [min, max, step]
                                    loguniform: [min, max, log_base]
                                    qloguniform: [min, max, step, log_base]
                                    choice: [each categorical values, ...]
                    # First format is recommaneded, but you can use any format.
                    # Please refer SingleSearchSpace class comment for
                    # detailed explanation of each vlaues.
                }
                "another_hyper_parameter_name" : {...}
                ...
            }
    """

    def __init__(
        self,
        search_space: dict[str, dict[str, Any]],
    ) -> None:
        self.search_space: dict[str, SingleSearchSpace] = {}

        try:
            for key, val in search_space.items():  # pylint: disable=too-many-nested-blocks
                self.search_space[key] = SingleSearchSpace(**val)
        except Exception:
            msg = f"Failed to create SingleSearchSpace. key={key}, val={val}"
            logging.exception(msg)
            raise

    def __getitem__(self, key: str) -> SingleSearchSpace:
        """Get search space by key."""
        try:
            return self.search_space[key]
        except KeyError as exc:
            error_msg = f"There is no search space named {key}."
            raise KeyError(error_msg) from exc

    def __repr__(self) -> str:
        """Print all search spaces."""
        return "\n".join(f"{key} => {val}" for key, val in self.search_space.items())

    def __iter__(self) -> Generator[str, Any, None]:
        """Iterate search spaces."""
        return self._search_space_generator()

    def __len__(self) -> int:
        """Number of search spaces."""
        return len(self.search_space)

    def _search_space_generator(self) -> Generator[str, Any, None]:
        yield from self.search_space

    def has_categorical_param(self) -> bool:
        """Check there is a search space whose type is choice."""
        return any(param.is_categorical() for param in self.search_space.values())

    def get_real_config(self, config: dict) -> dict:
        """Convert search space of each config from HPO perspective to human perspective.

        Args:
            config (dict): config to convert

        Returns:
            Dict: config converted to human perspective.
        """
        real_config = {}
        for param, value in config.items():
            real_config[param] = self[param].space_to_real(value)
        return real_config

    def get_space_config(self, config: dict) -> dict:
        """Convert search space of each config from human perspective to HPO perspective.

        Args:
            config (dict): config to convert

        Returns:
            Dict: config converted to human perspective.
        """
        space_config = {}
        for param, value in config.items():
            space_config[param] = self[param].real_to_space(value)
        return space_config

    def get_bayeopt_search_space(self) -> dict:
        """Return hyper parameter serach sapce as bayeopt library format."""
        bayesopt_space = {}
        for key, val in self.search_space.items():
            bayesopt_space[key] = (val.lower_space(), val.upper_space())

        return bayesopt_space

    def convert_from_zero_one_scale_to_real_space(self, config: dict) -> dict:
        """Convert search space of each config from zero one scale to human perspective.

        Args:
            config (dict): config to convert

        Returns:
            dict: config converted to human perspective.
        """
        for key, val in config.items():
            lower = self.search_space[key].lower_space()
            upper = self.search_space[key].upper_space()
            real_val = (upper - lower) * val + lower  # type: ignore[operator]
            config[key] = real_val

        return self.get_real_config(config)
