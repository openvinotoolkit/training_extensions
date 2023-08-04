import inspect
from typing import Any, Dict, Optional

from rich.console import Console
from rich.table import Table


class BaseRegistry:
    def __init__(self, name: str):
        self._name = name
        self._module_dict = dict()
        self._registry_dict = dict()

    def get(self, module_type: str):
        # Return Registry
        if module_type in self._registry_dict:
            return self._registry_dict[module_type]
        # The module_dict is the highest priority.
        if module_type in self._module_dict:
            return self._module_dict[module_type]
        # Search all registry
        for module in self._registry_dict.values():
            if module_type in module:
                return module.get(module_type)
        return None

    def __len__(self):
        # Copy from mmcv.utils.registry.Registry
        return len(self._module_dict)

    def __contains__(self, key):
        # Copy from mmcv.utils.registry.Registry
        return self.get(key) is not None

    def __repr__(self):
        # Evolved from mmengine
        table = Table(title=f"Registry of {self._name}")
        table.add_column("Type", justify="left", style="yellow")
        table.add_column("Names", justify="left", style="cyan")
        table.add_column("Objects", justify="left", style="green")

        for name, obj in sorted(self._module_dict.items()):
            table.add_row("Custom", name, str(obj))

        if hasattr(self, "_registry_dict"):
            for registry_key in self._registry_dict.keys():
                registry = self._registry_dict[registry_key]
                for name, obj in sorted(registry._module_dict.items()):
                    table.add_row(registry_key, name, str(obj))

        console = Console()
        with console.capture() as capture:
            console.print(table, end="")

        return capture.get()

    @property
    def name(self) -> str:
        return self._name

    @property
    def module_dict(self) -> Dict[Any, Any]:
        """Dictionary of registered module."""
        return self._module_dict

    @property
    def registry_dict(self) -> Dict[Any, Any]:
        """Dictionary of registries."""
        return self._registry_dict

    def register_module(self, type: Optional[str] = None, name: Optional[str] = None, module=None, force=False):
        # Copy from mmcv.utils.registry.Registry
        if not inspect.isclass(module) and not inspect.isfunction(module):
            raise TypeError("module must be a class or a function, " f"but got {type(module)}")

        if type is not None:
            if type not in self._registry_dict:
                self._registry_dict[type] = BaseRegistry(name=type)
            self._registry_dict[type].register_module(name=name, module=module)
        else:
            if name is None:
                name = module.__name__
            if isinstance(name, str):
                name = [name]
            for key in name:
                if not force and key in self._module_dict:
                    raise KeyError(f"{key} is already registered " f"in {self.name}")
                self._module_dict[key] = module
