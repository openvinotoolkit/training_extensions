# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import inspect
from typing import Dict, List, Optional, Union

import torch

from ..op import Operation


class OperationModule(torch.nn.Module):
    def __init__(
        self,
        op: Operation,
        dependent_ops: Union[List[Optional[Operation]], Dict[str, Optional[Operation]]],
    ):
        super().__init__()

        self.op = op
        self._dependent_ops = torch.nn.ModuleDict()

        spec = inspect.getfullargspec(op.forward)
        kwargs = spec.args[1:]

        self._dependents_with_defaults = []
        if spec.defaults:
            self._dependents_with_defaults = spec.args[-len(spec.defaults) :]

        if isinstance(dependent_ops, list):
            assert len(dependent_ops) == len(kwargs)
            for op, kwarg in zip(dependent_ops, kwargs):
                self._dependent_ops[kwarg] = op
        elif isinstance(dependent_ops, dict):
            for kwarg in kwargs:
                self._dependent_ops[kwarg] = dependent_ops[kwarg]
        else:
            raise NotImplementedError

    def forward(self, *args, **kwargs):
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

        return self.op(**inputs)

    @property
    def type(self):
        return self.op.type

    @property
    def version(self):
        return self.op.version

    @property
    def name(self):
        return self.op.name

    @property
    def shape(self):
        return self.op.shape

    @property
    def attrs(self):
        return self.op.attrs
