"""Implementation of the BaseRegistry class for registry pattern."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import inspect
from typing import Callable

from rich.console import Console
from rich.table import Table


class BaseRegistry:
    """Base class for registry pattern implementation."""

    def __init__(self, name: str) -> None:
        """Initialize a new instance of the Registry class.

        Args:
            name (str): The name of the registry.
        """
        self._name = name
        self._module_dict: dict = {}
        self._registry_dict: dict = {}

    def get(self, module_type: str) -> Callable | None:
        """Retrieve a module from the registry by its type.

        Args:
            module_type (str): The type of the module to retrieve.

        Returns:
            Optional[Callable]: The module if found, otherwise None.
        """
        # Return Registry
        if module_type in self._registry_dict:
            return self._registry_dict[module_type]
        # The module_dict is the highest priority.
        if module_type in self.module_dict:
            return self.module_dict[module_type]
        # Search all registry
        for module in self._registry_dict.values():
            if module_type in module:
                return module.get(module_type)
        return None

    def __len__(self) -> int:
        """Get the number of registered modules in the registry.

        Returns:
            int: The number of registered modules in the registry.
        """
        return len(self.module_dict)

    def __contains__(self, key: str) -> bool:
        """Check if the given key is in the registry.

        Args:
            key (str): The key to check.

        Returns:
            bool: True if the key is in the registry, False otherwise.
        """
        return self.get(key) is not None

    def __repr__(self) -> str:
        """Return a string representation of the Registry object.

        The string representation includes a table with the following columns:
        - Type: The type of the object (always "Custom" for this registry).
        - Names: The names of the objects in the registry.
        - Objects: The string representation of the objects in the registry.

        If the registry has any sub-registries, they are also included in the table,
        with their type being the name of the sub-registry.

        Returns:
            A string representation of the Registry object.
        """
        # Evolved from mmengine
        table = Table(title=f"Registry of {self._name}")
        table.add_column("Type", justify="left", style="yellow")
        table.add_column("Names", justify="left", style="cyan")
        table.add_column("Objects", justify="left", style="green")

        for name, obj in sorted(self.module_dict.items()):
            table.add_row("Custom", name, str(obj))

        if hasattr(self, "_registry_dict"):
            for registry_key in self._registry_dict:
                registry = self._registry_dict[registry_key]
                for name, obj in sorted(registry.module_dict.items()):
                    table.add_row(registry_key, name, str(obj))

        console = Console()
        with console.capture() as capture:
            console.print(table, end="")

        return capture.get()

    @property
    def name(self) -> str:
        """Returns the name of the registry."""
        return self._name

    @property
    def module_dict(self) -> dict:
        """Dictionary of registered module."""
        return self._module_dict

    @property
    def registry_dict(self) -> dict:
        """Dictionary of registries."""
        return self._registry_dict

    def register_module(
        self,
        type_name: str | None = None,
        name: str | None = None,
        module: type | Callable | None = None,
        force: bool = False,
    ) -> None:
        """Register a module to the registry.

        Args:
            type_name (str, optional): The name of the type to register the module under.
                If None, the module will be registered under its own name.
            name (str, optional): The name to register the module under. If None, the
                module's __name__ attribute will be used.
            module (type or callable, optional): The module to register.
            force (bool, optional): Whether to overwrite an existing module with the same name.

        Raises:
            TypeError: If the module is not a class or function.
            KeyError: If force is False and a module with the same name already exists.
        """
        # Copy from mmcv.utils.registry.Registry
        if not inspect.isclass(module) and not inspect.isfunction(module):
            msg = f"module must be a class or a function, but got {module!s}"
            raise TypeError(msg)

        if type_name is not None:
            if type_name not in self._registry_dict:
                self._registry_dict[type_name] = BaseRegistry(name=type_name)
            self._registry_dict[type_name].register_module(name=name, module=module)
        else:
            if name is None:
                name = module.__name__
            name_lst = [name]
            for key in name_lst:
                if not force and key in self._module_dict:
                    msg = f"{key} is already registered in {self.name}"
                    raise KeyError(msg)
                self._module_dict[key] = module
