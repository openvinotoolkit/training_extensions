# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""This implementation replaces the functionality of mmengine utils."""
# TODO(someone): Revisit mypy errors after deprecation of mmlab
# mypy: ignore-errors

from __future__ import annotations

import copy
import os
import re
from collections import OrderedDict, abc, namedtuple
from pathlib import Path
from typing import Any, Iterator, Union
from warnings import warn

import numpy as np
import torch
from torch import distributed as torch_dist
from torch import nn
from torch.utils.model_zoo import load_url


def get_dist_info() -> tuple[int, int]:
    """Get distributed information of the given process group.

    Note:
        Calling ``get_dist_info`` in non-distributed environment will return
        (0, 1).

    Returns:
        tuple[int, int]: Return a tuple containing the ``rank`` and
        ``world_size``.
    """
    if torch_dist.is_available() and torch_dist.is_initialized():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        group = torch_dist.distributed_c10d._get_default_group()  # noqa: SLF001
        world_size = torch_dist.get_world_size(group)
        rank = torch_dist.get_rank(group)
    else:
        world_size = 1
        rank = 0
    return rank, world_size


def load_checkpoint(
    model: nn.Module,
    checkpoint: str,
    map_location: str = "cpu",
    strict: bool = False,
    prefix: str = "",
) -> None:
    """Load state dict from path of checkpoint and dump to model."""
    if Path(checkpoint).exists():
        load_checkpoint_to_model(
            model,
            torch.load(checkpoint, map_location),
            strict=strict,
            prefix=prefix,
        )
    else:
        load_checkpoint_to_model(
            model,
            load_from_http(checkpoint, map_location),
            strict=strict,
            prefix=prefix,
        )


def load_from_http(
    filename: str,
    map_location: str | None = None,
    model_dir: Path | str | None = None,
    progress: bool = os.isatty(0),
) -> dict[str, Any]:
    """Loads a checkpoint from an HTTP URL.

    Copy of mmengine.runner.checkpoint.load_from_http.

    Args:
        filename (str): The URL of the checkpoint file.
        map_location (str | None, optional): Specifies where to load the checkpoint onto.
            Defaults to None.
        model_dir (str | None, optional): The directory to save the downloaded checkpoint.
            Defaults to None.
        progress (bool, optional): Whether to display a progress bar while downloading the checkpoint.
            Defaults to True if running in a terminal, otherwise False.

    Returns:
        dict[str, Any]: The loaded checkpoint.

    Raises:
        None

    """
    rank, world_size = get_dist_info()
    if rank == 0:
        checkpoint = load_url(filename, model_dir=model_dir, map_location=map_location, progress=progress)
    if world_size > 1:
        torch_dist.barrier()
        if rank > 0:
            checkpoint = load_url(filename, model_dir=model_dir, map_location=map_location, progress=progress)
    return checkpoint


class _IncompatibleKeys(namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return "<All keys matched successfully>"
        return super().__repr__()

    __str__ = __repr__


def load_state_dict(module: nn.Module, state_dict: OrderedDict, strict: bool = False) -> None:
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Defaults to False.
    """
    unexpected_keys: list[str] = []
    missing_keys: list[str] = []
    err_msg: list[str] = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata  # noqa: SLF001

    # use _load_from_state_dict to enable checkpoint version control
    def load(module: nn.Module, local_state_dict: dict, prefix: str = "") -> None:
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(  # noqa: SLF001
            local_state_dict,
            prefix,
            local_metadata,
            True,
            missing_keys,
            unexpected_keys,
            err_msg,
        )
        for name, child in module._modules.items():  # noqa: SLF001
            if child is not None:
                child_prefix = prefix + name + "."
                child_state_dict = {k: v for k, v in local_state_dict.items() if k.startswith(child_prefix)}
                load(child, child_state_dict, child_prefix)

        # Note that the hook can modify missing_keys and unexpected_keys.
        incompatible_keys = _IncompatibleKeys(missing_keys, unexpected_keys)
        if hasattr(module, "_load_state_dict_post_hooks"):
            for hook in module._load_state_dict_post_hooks.values():  # noqa: SLF001
                _ = hook(module, incompatible_keys)

    load(module, state_dict)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [key for key in missing_keys if "num_batches_tracked" not in key]

    if unexpected_keys:
        err_msg.append(f'unexpected key in source state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(0, "The model and loaded state dict do not match exactly\n")
        if strict:
            raise RuntimeError("\n".join(err_msg))
        warn("\n".join(err_msg), stacklevel=1)


def load_checkpoint_to_model(
    model: nn.Module,
    checkpoint: dict,
    strict: bool = False,
    prefix: str = "",
) -> None:
    """Loads a checkpoint dictionary into a PyTorch model.

    Copy of mmengine.runner.checkpoint._load_checkpoint_to_model.

    Args:
        model (nn.Module): The PyTorch model to load the checkpoint into.
        checkpoint (dict): The checkpoint dictionary containing the model's state_dict.
        strict (bool, optional): Whether to strictly enforce that the keys in the checkpoint match the keys
            in the model's state_dict. Defaults to False.

    Returns:
        None
    """
    # get state_dict from checkpoint
    # TODO(Kirill): remove this when RTDETR weights is updloaded to openvino storage.
    state_dict = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint.get("state_dict", checkpoint)

    # strip prefix of state_dict
    metadata = getattr(state_dict, "_metadata", OrderedDict())
    for p, r in [(r"^module\.", ""), (rf"^{prefix}\.", "")]:
        state_dict = OrderedDict({re.sub(p, r, k): v for k, v in state_dict.items()})

    # Keep metadata in state_dict
    state_dict._metadata = metadata  # noqa: SLF001

    # load state_dict
    load_state_dict(model, state_dict, strict)


def is_seq_of(
    seq: Any,  # noqa: ANN401
    expected_type: type | tuple,
    seq_type: type | None = None,
) -> bool:
    """Check whether it is a sequence of some type.

    Copied from mmengine.utils.misc.is_seq_of

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type or tuple): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type. Defaults to None.

    Returns:
        bool: Return True if ``seq`` is valid else False.

    Examples:
        >>> from mmengine.utils import is_seq_of
        >>> seq = ['a', 'b', 'c']
        >>> is_seq_of(seq, str)
        True
        >>> is_seq_of(seq, int)
        False
    """
    exp_seq_type = abc.Sequence if seq_type is None else seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    return all(isinstance(item, expected_type) for item in seq)


def is_tuple_of(seq: Any, expected_type: type | tuple) -> bool:  # noqa: ANN401
    """Check whether it is a tuple of some type.

    Copied from mmengine.utils.misc.is_tuple_of

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=tuple)


BoolTypeTensor = Union[torch.BoolTensor, torch.cuda.BoolTensor]
LongTypeTensor = Union[torch.LongTensor, torch.cuda.LongTensor]
IndexType = Union[str, slice, int, list, LongTypeTensor, BoolTypeTensor, np.ndarray]


class InstanceData:
    """A base data interface that supports Tensor-like and dict-like operations.

    This class is from https://github.com/open-mmlab/mmengine/blob/66fb81f7b392b2cd304fc1979d8af3cc71a011f5/mmengine/structures/instance_data.py
    and slightly modified.

    Args:
        metainfo (dict, optional): A dict contains the meta information
            of single image, such as ``dict(img_shape=(512, 512, 3),
            scale_factor=(1, 1, 1, 1))``. Defaults to None.
        kwargs (dict, optional): A dict contains annotations of single image or
            model predictions. Defaults to None.
    """

    def __init__(self, *, metainfo: dict | None = None, **kwargs) -> None:
        self._metainfo_fields: set = set()
        self._data_fields: set = set()

        if metainfo is not None:
            self.set_metainfo(metainfo=metainfo)
        if kwargs:
            self.set_data(kwargs)

    def set_metainfo(self, metainfo: dict) -> None:
        """Set or change key-value pairs in ``metainfo_field`` by parameter ``metainfo``.

        Args:
            metainfo (dict): A dict contains the meta information
                of image, such as ``img_shape``, ``scale_factor``, etc.
        """
        meta = copy.deepcopy(metainfo)
        for k, v in meta.items():
            self.set_field(name=k, value=v, field_type="metainfo", dtype=None)

    def set_data(self, data: dict) -> None:
        """Set or change key-value pairs in ``data_field`` by parameter ``data``.

        Args:
            data (dict): A dict contains annotations of image or
                model predictions.
        """
        for k, v in data.items():
            # Use `setattr()` rather than `self.set_field` to allow `set_data`
            # to set property method.
            setattr(self, k, v)

    def update(self, instance: InstanceData) -> None:
        """The method updates the InstanceData with the elements from another InstanceData object.

        Args:
            instance (InstanceData): Another InstanceData object for
                update the current object.
        """
        self.set_metainfo(dict(instance.metainfo_items()))
        self.set_data(dict(instance.items()))

    def new(self, *, metainfo: dict | None = None, **kwargs) -> InstanceData:
        """Return a new data element with same type.

        If ``metainfo`` and ``data`` are None, the new data element will have same metainfo and
        data. If metainfo or data is not None, the new result will overwrite it
        with the input value.

        Args:
            metainfo (dict, optional): A dict contains the meta information
                of image, such as ``img_shape``, ``scale_factor``, etc.
                Defaults to None.
            kwargs (dict): A dict contains annotations of image or
                model predictions.

        Returns:
            InstanceData: A new data element with same type.
        """
        new_data = self.__class__()

        if metainfo is not None:
            new_data.set_metainfo(metainfo)
        else:
            new_data.set_metainfo(dict(self.metainfo_items()))
        if kwargs:
            new_data.set_data(kwargs)
        else:
            new_data.set_data(dict(self.items()))
        return new_data

    def clone(self) -> InstanceData:
        """Deep copy the current data element.

        Returns:
            InstanceData: The copy of current data element.
        """
        clone_data = self.__class__()
        clone_data.set_metainfo(dict(self.metainfo_items()))
        clone_data.set_data(dict(self.items()))
        return clone_data

    def keys(self) -> list:
        """Returns lits contains all keys in data_fields."""
        private_keys = {"_" + key for key in self._data_fields if isinstance(getattr(type(self), key, None), property)}
        return list(self._data_fields - private_keys)

    def metainfo_keys(self) -> list:
        """Returns list contains all keys in metainfo_fields."""
        return list(self._metainfo_fields)

    def values(self) -> list:
        """Returns list contains all values in data."""
        return [getattr(self, k) for k in self.keys()]

    def metainfo_values(self) -> list:
        """Returns list contains all values in metainfo."""
        return [getattr(self, k) for k in self.metainfo_keys()]

    def all_keys(self) -> list:
        """Returns list contains all keys in metainfo and data."""
        return self.metainfo_keys() + self.keys()

    def all_values(self) -> list:
        """Returns list contains all values in metainfo and data."""
        return self.metainfo_values() + self.values()

    def all_items(self) -> Iterator[tuple[str, Any]]:
        """Returns iterator object whose element is (key, value) tuple pairs for ``metainfo`` and ``data``."""
        for k in self.all_keys():
            yield (k, getattr(self, k))

    def items(self) -> Iterator[tuple[str, Any]]:
        """Returns iterator object whose element is (key, value) tuple pairs for ``data``."""
        for k in self.keys():
            yield (k, getattr(self, k))

    def metainfo_items(self) -> Iterator[tuple[str, Any]]:
        """Returns iterator object whose element is (key, value) tuple pairs for ``metainfo``."""
        for k in self.metainfo_keys():
            yield (k, getattr(self, k))

    @property
    def metainfo(self) -> dict:
        """dict: A dict contains metainfo of current data element."""
        return dict(self.metainfo_items())

    def __setattr__(self, name: str, value: Any):  # noqa: ANN401
        """Setattr is only used to set data."""
        if name in ("_metainfo_fields", "_data_fields"):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                msg = f"{name} has been used as a private attribute, which is immutable."
                raise AttributeError(msg)
        else:
            self.set_field(name=name, value=value, field_type="data", dtype=None)

    __setitem__ = __setattr__

    def __getitem__(self, item: IndexType) -> InstanceData:
        """Get item mehod.

        Args:
            item (str, int, list, :obj:`slice`, :obj:`numpy.ndarray`,
                :obj:`torch.LongTensor`, :obj:`torch.BoolTensor`):
                Get the corresponding values according to item.

        Returns:
            :obj:`InstanceData`: Corresponding values.
        """
        if isinstance(item, list):
            item = np.array(item)
        if isinstance(item, np.ndarray):
            # The default int type of numpy is platform dependent, int32 for
            # windows and int64 for linux. `torch.Tensor` requires the index
            # should be int64, therefore we simply convert it to int64 here.
            # More details in https://github.com/numpy/numpy/issues/9464
            item = item.astype(np.int64) if item.dtype == np.int32 else item
            item = torch.from_numpy(item)

        if isinstance(item, str):
            return getattr(self, item)

        if isinstance(item, int):
            if item >= len(self) or item < -len(self):
                msg = f"Index {item} out of range!"
                raise IndexError(msg)
            item = slice(item, None, len(self))

        new_data = self.__class__(metainfo=self.metainfo)
        if isinstance(item, torch.Tensor):
            for k, v in self.items():
                if isinstance(v, torch.Tensor):
                    new_data[k] = v[item]
                elif isinstance(v, np.ndarray):
                    new_data[k] = v[item.cpu().numpy()]
                elif isinstance(v, (str, list, tuple)) or (hasattr(v, "__getitem__") and hasattr(v, "cat")):
                    # convert to indexes from BoolTensor
                    if isinstance(item, BoolTypeTensor.__args__):
                        indexes = torch.nonzero(item).view(-1).cpu().numpy().tolist()
                    else:
                        indexes = item.cpu().numpy().tolist()
                    slice_list = []
                    if indexes:
                        for index in indexes:
                            slice_list.append(slice(index, None, len(v)))  # noqa: PERF401
                    else:
                        slice_list.append(slice(None, 0, None))
                    r_list = [v[s] for s in slice_list]
                    if isinstance(v, (str, list, tuple)):
                        new_value = r_list[0]
                        for r in r_list[1:]:
                            new_value = new_value + r
                    else:
                        new_value = v.cat(r_list)
                    new_data[k] = new_value
                else:
                    msg = (
                        f"The type of `{k}` is `{type(v)}`, "
                        "which has no attribute of `cat`, so it does not support slice with `bool`"
                    )
                    raise ValueError(msg)

        else:
            # item is a slice
            for k, v in self.items():
                new_data[k] = v[item]
        return new_data

    def __delattr__(self, item: str):
        """Delete the item in dataelement.

        Args:
            item (str): The key to delete.
        """
        if item in ("_metainfo_fields", "_data_fields"):
            msg = f"{item} has been used as a private attribute, which is immutable."
            raise AttributeError(msg)
        super().__delattr__(item)
        if item in self._metainfo_fields:
            self._metainfo_fields.remove(item)
        elif item in self._data_fields:
            self._data_fields.remove(item)

    # dict-like methods
    __delitem__ = __delattr__

    def get(self, key: str, default: Any | None = None) -> Any:  # noqa: ANN401
        """Get property in data and metainfo as the same as python."""
        # Use `getattr()` rather than `self.__dict__.get()` to allow getting
        # properties.
        return getattr(self, key, default)

    def pop(self, *args) -> Any:  # noqa: ANN401
        """Pop property in data and metainfo as the same as python."""
        name = args[0]
        if name in self._metainfo_fields:
            self._metainfo_fields.remove(args[0])
            return self.__dict__.pop(*args)

        if name in self._data_fields:
            self._data_fields.remove(args[0])
            return self.__dict__.pop(*args)

        # with default value
        if len(args) == 2:
            return args[1]

        msg = f"{args[0]} is not contained in metainfo or data"
        raise KeyError(msg)

    def __contains__(self, item: str) -> bool:
        """Whether the item is in dataelement.

        Args:
            item (str): The key to inquire.
        """
        return item in self._data_fields or item in self._metainfo_fields

    def set_field(
        self,
        value: Any,  # noqa: ANN401
        name: str,
        dtype: type | tuple[type, ...] | None = None,
        field_type: str = "data",
    ) -> None:
        """Special method for set union field, used as property.setter functions."""
        if field_type == "metainfo":
            if name in self._data_fields:
                msg = f"Cannot set {name} to be a field of metainfo because {name} is already a data field"
                raise AttributeError(msg)
            self._metainfo_fields.add(name)
        else:
            if name in self._metainfo_fields:
                msg = f"Cannot set {name} to be a field of data because {name} is already a metainfo field"
                raise AttributeError(msg)
            self._data_fields.add(name)
        super().__setattr__(name, value)

    # Tensor-like methods
    def to(self, *args, **kwargs) -> InstanceData:
        """Apply same name function to all tensors in data_fields."""
        new_data = self.new()
        for k, v in self.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)  # noqa: PLW2901
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def cpu(self) -> InstanceData:
        """Convert all tensors to CPU in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, InstanceData)):
                v = v.cpu()  # noqa: PLW2901
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def cuda(self) -> InstanceData:
        """Convert all tensors to GPU in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, InstanceData)):
                v = v.cuda()  # noqa: PLW2901
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def detach(self) -> InstanceData:
        """Detach all tensors in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, InstanceData)):
                v = v.detach()  # noqa: PLW2901
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def numpy(self) -> InstanceData:
        """Convert all tensors to np.ndarray in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, InstanceData)):
                v = v.detach().cpu().numpy()  # noqa: PLW2901
                data = {k: v}
                new_data.set_data(data)
        return new_data

    def to_tensor(self) -> InstanceData:
        """Convert all np.ndarray to tensor in data."""
        new_data = self.new()
        for k, v in self.items():
            data = {}
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)  # noqa: PLW2901
                data[k] = v
            elif isinstance(v, InstanceData):
                v = v.to_tensor()  # noqa: PLW2901
                data[k] = v
            new_data.set_data(data)
        return new_data

    def to_dict(self) -> dict:
        """Convert InstanceData to dict."""
        return {k: v.to_dict() if isinstance(v, InstanceData) else v for k, v in self.all_items()}

    def __repr__(self) -> str:
        """Represent the object."""

        def _addindent(s_: str, num_spaces: int) -> str:
            """This func is modified from `pytorch`.

            https://github.com/pytorch/
            pytorch/blob/b17b2b1cc7b017c3daaeff8cc7ec0f514d42ec37/torch/nn/modu
            les/module.py#L29.

            Args:
                s_ (str): The string to add spaces.
                num_spaces (int): The num of space to add.

            Returns:
                str: The string after add indent.
            """
            s = s_.split("\n")
            # don't do anything for single-line stuff
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            return first + "\n" + s

        def dump(obj: Any) -> str:  # noqa: ANN401
            """Represent the object.

            Args:
                obj (Any): The obj to represent.

            Returns:
                str: The represented str.
            """
            _repr = ""
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _repr += f"\n{k}: {_addindent(dump(v), 4)}"
            elif isinstance(obj, InstanceData):
                _repr += "\n\n    META INFORMATION"
                metainfo_items = dict(obj.metainfo_items())
                _repr += _addindent(dump(metainfo_items), 4)
                _repr += "\n\n    DATA FIELDS"
                items = dict(obj.items())
                _repr += _addindent(dump(items), 4)
                classname = obj.__class__.__name__
                _repr = f"<{classname}({_repr}\n) at {hex(id(obj))}>"
            else:
                _repr += repr(obj)
            return _repr

        return dump(self)

    def __len__(self) -> int:
        """int: The length of InstanceData."""
        if len(self._data_fields) > 0:
            return len(self.values()[0])
        return 0
