"""Utils for dynamically importing stuff."""

# Copyright (C) 2021 Intel Corporation
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
from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Callable, TypeVar


def get_impl_class(impl_path: str) -> TypeVar:
    """Given a fully qualified path to a class, returns the class object.

    Args:
        impl_path (str): The fully qualified path to the class.

    Returns:
        The class object.

    Raises:
        Exception: If the class cannot be imported.
    """
    task_impl_module = None
    result = None
    try:
        task_impl_module_name, task_impl_class_name = impl_path.rsplit(".", 1)
        task_impl_module = importlib.import_module(task_impl_module_name)
        result = getattr(task_impl_module, task_impl_class_name)

    except Exception as e:
        if hasattr(task_impl_module, "DEBUG"):
            exception = getattr(task_impl_module, "DEBUG", None)
            if isinstance(exception, Exception):
                raise exception from e
        raise e from None
    return result


def get_default_args(func: Callable) -> list:
    """Return a list of tuples containing the names and default values of the non-keyword arguments of a function.

    Args:
        func (Callable): The function to inspect.

    Returns:
        list: A list of tuples containing the names and default values of the non-keyword arguments of the function.
    """
    signature = inspect.signature(func)
    non_default_args = []

    for name, parameter in signature.parameters.items():
        if parameter.default is not inspect.Parameter.empty and parameter.default is not None:
            non_default_args.append((name, parameter.default))

    return non_default_args


def get_all_args(func: Callable) -> list[str]:
    """Return a list of all argument names for a given function.

    Args:
        func (Callable): The function to inspect.

    Returns:
        List[str]: A list of argument names.
    """
    signature = inspect.signature(func)
    return list(signature.parameters.keys())


def get_otx_root_path() -> str:
    """Return the root path of the otx module.

    Returns:
        str: The root path of the otx module.

    Raises:
        ModuleNotFoundError: If the otx module is not found.
    """
    otx_module = importlib.import_module("otx")
    if otx_module:
        file_path = inspect.getfile(otx_module)
        return str(Path(file_path).parent)
    msg = "Cannot found otx."
    raise ModuleNotFoundError(msg)


def get_files_dict(folder_path: str | Path) -> dict[str, str]:
    """Return a dictionary containing the names and paths of all files in the specified folder.

    Args:
        folder_path (Union[str, Path]): The path to the folder containing the files.

    Returns:
        Dict[str, str]: A dictionary containing the names and paths of all files in the specified folder.
    """
    file_path_dict = {}

    _folder_path: Path = Path(folder_path)
    if not _folder_path.exists():
        msg = "The specified folder path does not exist."
        raise ValueError(msg)

    for file_path in _folder_path.iterdir():
        if file_path.is_file():
            file_name = file_path.stem
            file_path_dict[file_name] = str(file_path)

    return file_path_dict
