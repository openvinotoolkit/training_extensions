import inspect


class BaseRegistry:
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def get(self, module_type: str):
        raise NotImplementedError

    def __len__(self):
        # Copy from mmcv.utils.registry.Registry
        return len(self._module_dict)

    def __contains__(self, key):
        # Copy from mmcv.utils.registry.Registry
        return self.get(key) is not None

    def __repr__(self):
        # Copy from mmcv.utils.registry.Registry
        format_str = self.__class__.__name__ + f"(name={self._name}, " f"items={self._module_dict})"
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        # Copy from mmcv.utils.registry.Registry
        return self._module_dict

    def register_module(self, name=None, module=None, force=False):
        # Copy from mmcv.utils.registry.Registry
        if not inspect.isclass(module) and not inspect.isfunction(module):
            raise TypeError("module must be a class or a function, " f"but got {type(module)}")

        if name is None:
            name = module.__name__
        if isinstance(name, str):
            name = [name]
        for key in name:
            if not force and key in self._module_dict:
                raise KeyError(f"{key} is already registered " f"in {self.name}")
            self._module_dict[key] = module
