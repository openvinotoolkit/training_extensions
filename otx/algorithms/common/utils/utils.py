"""Collections of Utils for common OTX algorithms."""

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

import importlib
import inspect
from collections import defaultdict
from typing import Callable, Optional, Tuple

import yaml

from otx.api.utils.argument_checks import YamlFilePathCheck, check_input_parameters_type


class UncopiableDefaultDict(defaultdict):
    """Defauldict type object to avoid deepcopy."""

    def __deepcopy__(self, memo):
        """Deepcopy."""
        return self


@check_input_parameters_type({"path": YamlFilePathCheck})
def load_template(path):
    """Loading model template function."""
    with open(path, encoding="UTF-8") as f:
        template = yaml.safe_load(f)
    return template


@check_input_parameters_type()
def get_task_class(path: str):
    """Return Task classes."""
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


@check_input_parameters_type()
def get_arg_spec(  # noqa: C901  # pylint: disable=too-many-branches
    fn: Callable,  # pylint: disable=invalid-name
    depth: Optional[int] = None,
) -> Tuple[str, ...]:
    """Get argument spec of function."""

    args = set()

    cls_obj = None
    if inspect.ismethod(fn):
        fn_name = fn.__name__
        cls_obj = fn.__self__
        if not inspect.isclass(cls_obj):
            cls_obj = cls_obj.__class__
    else:
        fn_name = fn.__name__
        names = fn.__qualname__.split(".")
        if len(names) > 1 and names[-1] == fn_name:
            cls_obj = globals()[".".join(names[:-1])]

    if cls_obj:
        for obj in cls_obj.mro():  # type: ignore
            fn_obj = cls_obj.__dict__.get(fn_name, None)
            if fn_obj is not None:
                if isinstance(fn_obj, staticmethod):
                    cls_obj = None
                    break

    if cls_obj is None:
        # function, staticmethod
        spec = inspect.getfullargspec(fn)
        args.update(spec.args)
    else:
        # method, classmethod
        for i, obj in enumerate(cls_obj.mro()):  # type: ignore
            if depth is not None and i == depth:
                break
            method = getattr(obj, fn_name, None)
            if method is None:
                break
            spec = inspect.getfullargspec(method)
            args.update(spec.args[1:])
            if spec.varkw is None and spec.varargs is None:
                break
    return tuple(args)
