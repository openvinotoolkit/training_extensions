"""Search space class for HPO."""

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


import math
import typing
from typing import Any, Dict, List, Optional, Tuple, Union

from otx.hpo.utils import check_positive
from otx.utils.logger import get_logger

logger = get_logger()

AVAILABLE_SEARCH_SPACE_TYPE = ["uniform", "quniform", "loguniform", "qloguniform", "choice"]


class SingleSearchSpace:
    """The class which implements single search space used for HPO.

    This class supports uniform and quantized uniform with normal and log scale
    in addition to categorical type. Quantized type has step which is unit for change.

    Args:
        type (str): type of hyper parameter in search space.
                    supported types: uniform, loguniform, quniform, qloguniform, choice
        min (float or int, optional): upper bounding of search space.
                                      If type isn't choice, this value is required.
        max (float or int, optional): lower bounding of search space
                                      If type isn't choice, this value is required.
        step (int, optional): unit for change. If type is quniform or qloguniform,
                              This value is required.
        log_base (int, optional): base of logarithm. Default value is 2.
        choice_list (list, optional): candidiates for choice type. If task is choice,
                                      this value is required.
    """

    def __init__(
        self,
        type: str,
        min: Optional[Union[float, int]] = None,
        max: Optional[Union[float, int]] = None,
        step: Optional[Union[float, int]] = None,
        log_base: Optional[int] = 2,
        choice_list: Optional[Union[List, Tuple]] = None,
    ):
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
    def type(self):
        """Type of hyper parameter in search space."""
        return self._type

    @property
    def min(self):
        """Lower bounding of search space."""
        return self._min

    @property
    def max(self):
        """Upper bounding of search space."""
        return self._max

    @property
    def step(self):
        """Unit for change."""
        return self._step

    @property
    def log_base(self):
        """Base of logarithm."""
        return self._log_base

    @property
    def choice_list(self):
        """Candidiates for choice type."""
        return self._choice_list

    def set_value(
        self,
        type: Optional[str] = None,
        min: Optional[Union[float, int]] = None,
        max: Optional[Union[float, int]] = None,
        step: Optional[Union[float, int]] = None,
        log_base: Optional[int] = None,
        choice_list: Optional[Union[List, Tuple]] = None,
    ):
        # pylint: disable=redefined-builtin
        """Set attributes of the class.

        There are some dependencies when setting attributes of this class.
        So, modifying a single attribute at a time can set wrong value.
        For example, `max` can be set as a value lower than `min`.
        To prevent it, this function priovides a way to set all necessary values at a time.

        Args:
            type (str): type of hyper parameter in search space.
                        supported types: uniform, loguniform, quniform, qloguniform, choice
            min (float or int, optional): upper bounding of search space.
                                        If type isn't choice, this value is required.
            max (float or int, optional): lower bounding of search space
                                        If type isn't choice, this value is required.
            step (int, optional): unit for change. If type is quniform or qloguniform,
                                This value is required.
            log_base (int, optional): base of logarithm. Default value is 2.
            choice_list (list, optional): candidiates for choice type. If task is choice,
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

    def _align_min_max_to_choice_list_if_categorical(self):
        if self.is_categorical():
            self._min = 0
            self._max = len(self._choice_list) - 1

    def _check_all_value_is_right(self):
        # pylint: disable=too-many-branches
        if self._type not in AVAILABLE_SEARCH_SPACE_TYPE:
            raise ValueError(
                f"type should be one of {', '.join(AVAILABLE_SEARCH_SPACE_TYPE)}. " f"But your argument is {self._type}"
            )

        if self.is_categorical():
            if len(self._choice_list) <= 1:
                raise ValueError("If type is choice, choice_list should have more than one element")
            if self._min != 0:
                raise ValueError("if type is categorical, min should be 0.")
            if self._max != len(self._choice_list) - 1:
                raise ValueError("if type is categorical, max should be last index number of choice_list.")
        else:
            if min is None:
                raise ValueError("If type isn't choice, you should set min value of search space.")
            if max is None:
                raise ValueError("If type isn't choice, you should set max value of search space.")

            if self._min >= self._max:
                raise ValueError(
                    "max value should be greater than min value.\n" f"max value : {self._max} / min value : {self._min}"
                )

            if self.use_log_scale():
                if self._log_base <= 1:
                    raise ValueError("log base should be greater than 1.\n" f"your log base value is {self._log_base}.")
                if self._min <= 0:
                    raise ValueError(
                        "If you use log scale, min value should be greater than 0.\n" f"your min value is {self._min}"
                    )
            if self.use_quantized_step():
                if self._step is None:
                    raise ValueError(f"The {self._type} type requires step value. But it doesn't exists")
                check_positive(self._step, "step")
                if self._step > self._max - self._min:
                    raise ValueError(
                        "Difference between min and max is greater than step.\n"
                        f"Current value is min : {self._min}, max : {self._max}, step : {self._step}"
                    )

    def __repr__(self):
        """Print serach space status."""
        if self.is_categorical():
            return f"type: {self._type}, candidiate : {', '.join(self._choice_list)}"
        rep = f"type: {self._type}, search space : {self._min} ~ {self._max}"
        if self.use_quantized_step():
            rep += f", step : {self._step}"
        if self.use_log_scale():
            rep += f", log base : {self._log_base}"
        return rep

    def is_categorical(self):
        """Check current instance is categorical type."""
        return self._type == "choice"

    def use_quantized_step(self):
        """Check current instance is one of type to use `step`."""
        return self._type in ("quniform", "qloguniform")

    def use_log_scale(self):
        """Check current instance is one of type to use `log scale`."""
        return self._type in ("loguniform", "qloguniform")

    def lower_space(self):
        """Get lower bound value considering log scale if necessary."""
        if self.use_log_scale():
            return math.log(self._min, self._log_base)
        return self._min

    def upper_space(self):
        """Get upper bound value considering log scale if necessary."""
        if self.use_log_scale():
            return math.log(self._max, self._log_base)
        return self._max

    @typing.no_type_check
    def space_to_real(self, number: Union[int, float]) -> Union[int, float]:
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

    def real_to_space(self, number: Union[int, float]) -> Union[int, float]:
        """Convert search space from human perspective to HPO perspective.

        Args:
            number (Union[int, float]): Value to convert

        Returns:
            Union[int, float]: value converted to HPO perspective.
        """
        if self.use_log_scale():
            return math.log(number, self._log_base)  # type: ignore
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
                    "param_type": type of search space of hyper parameter.
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
        search_space: Dict[str, Dict[str, Any]],
    ):
        self.search_space: Dict[str, SingleSearchSpace] = {}

        for key, val in search_space.items():  # pylint: disable=too-many-nested-blocks
            if "range" not in val:
                val["type"] = val.pop("param_type")
                self.search_space[key] = SingleSearchSpace(**val)
            else:
                args = {"type": val["param_type"]}
                if val["param_type"] == "choice":
                    args["choice_list"] = val["range"]
                else:
                    if len(val) != 2:
                        logger.warning("If there is the range in keys, then other values are ignored.")
                    try:
                        args["min"] = val["range"][0]
                        args["max"] = val["range"][1]
                        if args["type"] == "quniform":
                            args["step"] = val["range"][2]
                        elif args["type"] == "loguniform":
                            if len(val["range"]) == 3:
                                args["log_base"] = val["range"][2]
                        elif args["type"] == "qloguniform":
                            args["step"] = val["range"][2]
                            if len(val["range"]) == 4:
                                args["log_base"] = val["range"][3]
                    except IndexError as exc:
                        raise ValueError(
                            "You should give all necessary value depending on search space type."
                            "which values are needed depending on type are as bellow."
                            "   - uniform : min value, max value"
                            "   - quniform : min value, max value, step"
                            "   - loguniform : min value, max value, log base(default 2)"
                            "   - qloguniform : min value, max value, step, log baes(default 2)"
                            "But your value is:"
                            f"  - {val['param_type']} : {', '.join([str(element) for element in val['range']])}"
                        ) from exc
                self.search_space[key] = SingleSearchSpace(**args)

    def __getitem__(self, key):
        """Get search space by key."""
        try:
            return self.search_space[key]
        except KeyError as exc:
            raise KeyError(f"There is no search space named {key}.") from exc

    def __repr__(self):
        """Print all search spaces."""
        return "\n".join(f"{key} => {val}" for key, val in self.search_space.items())

    def __iter__(self):
        """Iterate search spaces."""
        return self._search_space_generator()

    def __len__(self):
        """Number of search spaces."""
        return len(self.search_space)

    def _search_space_generator(self):
        for key in self.search_space:
            yield key

    def has_categorical_param(self):
        """Check there is a search space whose type is choice."""
        for param in self.search_space.values():
            if param.is_categorical():
                return True
        return False

    def get_real_config(self, config: Dict) -> Dict:
        """Convert search space of each config from HPO perspective to human perspective.

        Args:
            config (Dict): config to convert

        Returns:
            Dict: config converted to human perspective.
        """
        real_config = {}
        for param, value in config.items():
            real_config[param] = self[param].space_to_real(value)
        return real_config

    def get_space_config(self, config: Dict) -> Dict:
        """Convert search space of each config from human perspective to HPO perspective.

        Args:
            config (Dict): config to convert

        Returns:
            Dict: config converted to human perspective.
        """
        space_config = {}
        for param, value in config.items():
            space_config[param] = self[param].real_to_space(value)
        return space_config

    def get_bayeopt_search_space(self) -> Dict:
        """Return hyper parameter serach sapce as bayeopt library format."""
        bayesopt_space = {}
        for key, val in self.search_space.items():
            bayesopt_space[key] = (val.lower_space(), val.upper_space())

        return bayesopt_space

    def convert_from_zero_one_scale_to_real_space(self, config: Dict) -> Dict:
        """Convert search space of each config from zero one scale to human perspective.

        Args:
            config (Dict): config to convert

        Returns:
            Dict: config converted to human perspective.
        """
        for key, val in config.items():
            lower = self.search_space[key].lower_space()
            upper = self.search_space[key].upper_space()
            val = (upper - lower) * val + lower
            config[key] = val

        return self.get_real_config(config)
