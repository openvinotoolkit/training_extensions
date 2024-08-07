# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX utility functions."""

from __future__ import annotations

import importlib
import inspect
import pickle  # nosec B403 used pickle for internal state dump/load
from decimal import Decimal
from functools import partial
from types import LambdaType
from typing import TYPE_CHECKING, Any, Callable

import torch

from otx.core.model.base import OTXModel

if TYPE_CHECKING:
    from pathlib import Path

    from jsonargparse import Namespace


XPU_AVAILABLE = None
try:
    import intel_extension_for_pytorch  # noqa: F401
except ImportError:
    XPU_AVAILABLE = False


def get_using_dot_delimited_key(key: str, target: Any) -> Any:  # noqa: ANN401
    """Get values of attribute in target object using dot delimited key.

    For example, if key is "a.b.c", then get a value of 'target.a.b.c'.
    Target should be object having attributes, dictionary or list.
    To get an element in a list, an integer that is the index of corresponding value can be set as a key.

    Args:
        key (str): dot delimited key.
        val (Any): value to set.
        target (Any): target to set value to.
    """
    splited_key = key.split(".")
    for each_key in splited_key:
        if isinstance(target, dict):
            target = target[each_key]
        elif isinstance(target, list):
            if not each_key.isdigit():
                error_msg = f"Key should be integer but '{each_key}'."
                raise ValueError(error_msg)
            target = target[int(each_key)]
        else:
            target = getattr(target, each_key)
    return target


def set_using_dot_delimited_key(key: str, val: Any, target: Any) -> None:  # noqa: ANN401
    """Set values to attribute in target object using dot delimited key.

    For example, if key is "a.b.c", then value is set at 'target.a.b.c'.
    Target should be object having attributes, dictionary or list.
    To get an element in a list, an integer that is the index of corresponding value can be set as a key.

    Args:
        key (str): dot delimited key.
        val (Any): value to set.
        target (Any): target to set value to.
    """
    splited_key = key.split(".")
    for each_key in splited_key[:-1]:
        if isinstance(target, dict):
            target = target[each_key]
        elif isinstance(target, list):
            if not each_key.isdigit():
                error_msg = f"Key should be integer but '{each_key}'."
                raise ValueError(error_msg)
            target = target[int(each_key)]
        else:
            target = getattr(target, each_key)

    if isinstance(target, dict):
        target[splited_key[-1]] = val
    elif isinstance(target, list):
        if not splited_key[-1].isdigit():
            error_msg = f"Key should be integer but '{splited_key[-1]}'."
            raise ValueError(error_msg)
        target[int(splited_key[-1])] = val
    else:
        setattr(target, splited_key[-1], val)


def get_decimal_point(num: int | float) -> int:
    """Find a decimal point from the given float.

    Args:
        num (int | float): float to find a decimal point from.

    Returns:
        int: decimal point.
    """
    if isinstance((exponent := Decimal(str(num)).as_tuple().exponent), int):
        return abs(exponent)
    error_msg = f"Can't get an exponent from {num}."
    raise ValueError(error_msg)


def find_file_recursively(directory: Path, file_name: str) -> Path | None:
    """Find the file from the direcotry recursively. If multiple files have a same name, return one of them.

    Args:
        directory (Path): directory where to find.
        file_name (str): file name to find.

    Returns:
        Path | None: Found file. If it's failed to find a file, return None.
    """
    if found_file := list(directory.rglob(file_name)):
        return found_file[0]
    return None


def remove_matched_files(directory: Path, pattern: str, file_to_leave: Path | None = None) -> None:
    """Remove all files matched to pattern except file_to_leave.

    Args:
        directory (Path): direcetory to find files to remove.
        pattern (str): pattern to match a file name.
        file_not_to_remove (Path | None, optional): files to leave. Defaults to None.
    """
    for weight in directory.rglob(pattern):
        if weight != file_to_leave:
            weight.unlink()


def is_xpu_available() -> bool:
    """Checks if XPU device is available."""
    global XPU_AVAILABLE  # noqa: PLW0603
    if XPU_AVAILABLE is None:
        XPU_AVAILABLE = hasattr(torch, "xpu") and torch.xpu.is_available()
    return XPU_AVAILABLE


def get_model_cls_from_config(model_config: Namespace) -> type[OTXModel]:
    """Get Python model class from jsonargparse Namespace."""
    splited = model_config.class_path.split(".")
    module_path, class_name = ".".join(splited[:-1]), splited[-1]
    module = importlib.import_module(module_path)
    model_cls = getattr(module, class_name)

    if not issubclass(model_cls, OTXModel):
        raise TypeError(model_cls)

    return model_cls


def should_pass_label_info(model_cls: type[OTXModel]) -> bool:
    """Determine if label_info should be passed when instantiating the given model class.

    Args:
        model_cls (Type[OTXModel]): OTX model class to instantiate.

    Returns:
        bool: True if label_info should be passed, False otherwise.
    """
    label_info_param = inspect.signature(model_cls).parameters.get("label_info")
    return label_info_param is not None and label_info_param.default == label_info_param.empty


def can_pass_tile_config(model_cls: type[OTXModel]) -> bool:
    """Determine if tile_config can be passed when instantiating the given model class.

    Args:
        model_cls (Type[OTXModel]): OTX model class to instantiate.

    Returns:
        bool: True if tile_config can be passed, False otherwise.
    """
    tile_config_param = inspect.signature(model_cls).parameters.get("tile_config")
    return tile_config_param is not None


def get_class_initial_arguments() -> tuple:
    """Return arguments of class initilization. This function should be called in '__init__' function.

    Returns:
        tuple: class arguments.
    """
    keywords, _, _, values = inspect.getargvalues(inspect.stack()[1].frame)
    return tuple(values[key] for key in keywords[1:])


def find_unpickleable_obj(obj: Any, obj_name: str) -> list[str]:  # noqa: ANN401
    """Find which objects in 'obj' can't be pickled.

    Args:
        obj (Any): Object where to find unpickleable object.
        obj_name (str): Name of obj.

    Returns:
        list[str]: List of name of unpikcleable objects.
    """
    unpickleable_obj: dict[str, Any] = {}
    _find_unpickleable_obj(obj, obj_name, unpickleable_obj)

    if not unpickleable_obj:
        return []

    # get actual cause of unpickleable
    unpickleable_obj_keys = sorted(unpickleable_obj.keys())
    unpickleable_cause = [
        unpickleable_obj_keys[i]
        for i in range(len(unpickleable_obj_keys) - 1)
        if unpickleable_obj_keys[i] not in unpickleable_obj_keys[i + 1]
    ]
    unpickleable_cause.append(unpickleable_obj_keys[-1])

    return unpickleable_cause


def _find_unpickleable_obj(obj: Any, obj_name: str, unpickleable_obj: dict[str, Any]) -> None:  # noqa: ANN401
    if check_pickleable(obj):
        return

    unpickleable_obj[obj_name] = obj

    def _need_skip(obj: Any) -> bool:  # noqa: ANN401
        return isinstance(obj, memoryview)  # it makes core dumped

    def _make_iter(obj: Any) -> list[tuple[str, Any]]:  # noqa: ANN401
        if isinstance(obj, (list, tuple)):
            return [(f"[{i}]", obj[i]) for i in range(len(obj))]
        if isinstance(obj, dict):
            return [(f'["{key}"]', obj[key]) for key in obj]

        res = []
        for attr in dir(obj):
            if attr.startswith("__") and attr.endswith("__"):  # skip magic method
                continue
            try:
                attr_obj = getattr(obj, attr)
            except Exception:  # noqa: S112
                continue
            if callable(attr_obj) and not isinstance(attr_obj, (LambdaType, partial)):
                continue
            res.append((f".{attr}", attr_obj))
        return res

    for key, val in _make_iter(obj):
        if not _need_skip(val) and val not in unpickleable_obj.values():
            _find_unpickleable_obj(val, obj_name + key, unpickleable_obj)


def check_pickleable(obj: Any) -> bool:  # noqa: ANN401
    """Check object can be pickled."""
    try:
        pickled_data = pickle.dumps(obj)
        pickle.loads(pickled_data)  # noqa: S301 # nosec B301 used pickle for internal state dump/load
    except Exception:
        return False
    return True


def measure_flops(
    model: torch.nn.Module,
    forward_fn: Callable[[], torch.Tensor],
    loss_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    print_stats_depth: int = 0,
) -> int:
    """Utility to compute the total number of FLOPs used by a module during training or during inference."""
    from torch.utils.flop_counter import FlopCounterMode

    flop_counter = FlopCounterMode(model, display=print_stats_depth > 0, depth=print_stats_depth)
    with flop_counter:
        if loss_fn is None:
            forward_fn()
        else:
            loss_fn(forward_fn()).backward()
    return flop_counter.get_total_flops()
