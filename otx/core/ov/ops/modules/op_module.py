"""Operation module for otx.core.ov.ops.modeuls."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import inspect
from typing import Dict, List, Optional, Union

import torch

from ..op import Operation


class OperationModule(torch.nn.Module):
    """OperationModule class."""

    def __init__(
        self,
        op_v: Operation,
        dependent_ops: Union[List[Operation], Dict[str, Optional[Operation]]],
    ):
        super().__init__()

        self.op_v = op_v
        self._dependent_ops = torch.nn.ModuleDict()

        spec = inspect.getfullargspec(op_v.forward)
        kwargs = spec.args[1:]

        self._dependents_with_defaults = []
        if spec.defaults:
            self._dependents_with_defaults = spec.args[-len(spec.defaults) :]

        if isinstance(dependent_ops, list):
            assert len(dependent_ops) == len(kwargs)
            for op_, kwarg in zip(dependent_ops, kwargs):
                self._dependent_ops[kwarg] = op_
        elif isinstance(dependent_ops, dict):
            for kwarg in kwargs:
                self._dependent_ops[kwarg] = dependent_ops[kwarg]
        else:
            raise NotImplementedError

    def forward(self, *args, **kwargs):
        """Operationmodule's forward function."""
        inputs = {k: v() if v is not None else None for k, v in self._dependent_ops.items()}

        if args:
            empty_input_keys = [k for k, v in self._dependent_ops.items() if v is None]
            for key, val in zip(empty_input_keys, args):
                inputs[key] = val
        if kwargs:
            for key, val in kwargs.items():
                if inputs[key] is not None:
                    raise ValueError(f"duplicated key {key}")
                inputs[key] = val

        assert all(v is not None for v in inputs.values() if v not in self._dependents_with_defaults)

        return self.op_v(**inputs)

    @property
    def type(self):  # pylint: disable=invalid-overridden-method
        """Operationmodule's type property."""
        return self.op_v.type

    @property
    def version(self):
        """Operationmodule's version property."""
        return self.op_v.version

    @property
    def name(self):
        """Operationmodule's name property."""
        return self.op_v.name

    @property
    def shape(self):
        """Operationmodule's shape property."""
        return self.op_v.shape

    @property
    def attrs(self):
        """Operationmodule's attrs property."""
        return self.op_v.attrs
