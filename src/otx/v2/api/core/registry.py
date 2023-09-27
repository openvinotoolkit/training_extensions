import inspect
from typing import Callable, Optional, Type, Union

from rich.console import Console
from rich.table import Table


class BaseRegistry:
    def __init__(self, name: str) -> None:
        self._name = name
        self._module_dict: dict = {}
        self._registry_dict: dict = {}

    def get(self, module_type: str) -> Optional[Callable]:
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
        # Copy from mmcv.utils.registry.Registry
        return len(self.module_dict)

    def __contains__(self, key: str) -> bool:
        # Copy from mmcv.utils.registry.Registry
        return self.get(key) is not None

    def __repr__(self) -> str:
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
        type_name: Optional[str] = None,
        name: Optional[str] = None,
        module: Optional[Union[Type, Callable]] = None,
        force: bool = False,
    ) -> None:
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
