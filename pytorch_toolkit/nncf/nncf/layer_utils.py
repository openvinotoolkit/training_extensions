import torch.nn as nn
from .registry import Registry


COMPRESSION_MODULES = Registry('compression modules')


class ProxyModule:
    def __init__(self, module):
        self._module = module

    def __getattr__(self, name):
        return getattr(self._module, name)


class _NNCFModuleMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_ops = nn.ModuleDict()
        self.post_ops = nn.ModuleDict()

    def get_pre_op(self, key):
        return self.pre_ops[key]

    def get_post_op(self, key):
        return self.post_ops[key]

    def register_pre_forward_operation(self, op):
        key = str(len(self.pre_ops))
        self.pre_ops[key] = op
        return key

    def remove_pre_forward_operation(self, key):
        return self.pre_ops.pop(key)

    def register_post_forward_operation(self, op):
        key = str(len(self.post_ops))
        self.post_ops[key] = op
        return key

    def remove_post_forward_operation(self, key):
        return self.post_ops.pop(key)

    def forward(self, *args):
        proxy_module = ProxyModule(self)
        for op in self.pre_ops.values():
            op_args = op(proxy_module, args)
            if op_args is not None:
                if not isinstance(op_args, tuple):
                    op_args = tuple([op_args])
                args = op_args
        results = super().forward.__func__(proxy_module, *args)
        for op in self.post_ops.values():
            op_results = op(proxy_module, results)
            if op_results is not None:
                results = op_results
        return results
