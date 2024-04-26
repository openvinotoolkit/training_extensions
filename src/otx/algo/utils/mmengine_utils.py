# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""This implementation replaces the functionality of mmengine utils."""
# TODO(someone): Revisit mypy errors after deprecation of mmlab
# mypy: ignore-errors

from __future__ import annotations

import math
import os
import re
from collections import OrderedDict, abc, namedtuple
from typing import Any
from warnings import warn

from torch import Tensor, nn, no_grad
from torch import distributed as torch_dist
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


def load_from_http(
    filename: str,
    map_location: str | None = None,
    model_dir: str | None = None,
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
    state_dict = checkpoint.get("state_dict", checkpoint)

    # strip prefix of state_dict
    metadata = getattr(state_dict, "_metadata", OrderedDict())
    for p, r in [(r"^module\.", ""), (rf"^{prefix}\.", "")]:
        state_dict = OrderedDict({re.sub(p, r, k): v for k, v in state_dict.items()})

    # Keep metadata in state_dict
    state_dict._metadata = metadata  # noqa: SLF001

    # load state_dict
    load_state_dict(model, state_dict, strict)


def normal_init(module: nn.Module, mean: float = 0, std: float = 1, bias: float = 0) -> None:
    """Initialize the weights and biases of a module using a normal distribution.

    Copied from mmengine.model.weight_init.normal_init

    Args:
        module (nn.Module): The module to initialize.
        mean (float): The mean of the normal distribution. Default is 0.
        std (float): The standard deviation of the normal distribution. Default is 1.
        bias (float): The bias value. Default is 0.
    """
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module: nn.Module, val: float, bias: float = 0) -> None:
    """Initialize the weights and biases of a module with constant values.

    Copied from mmengine.model.weight_init.constant_init

    Args:
        module (nn.Module): The module to initialize.
        val (float): The constant value to initialize the weights with.
        bias (float, optional): The constant value to initialize the biases with. Defaults to 0.
    """
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(
    module: nn.Module,
    a: float = 0,
    mode: str = "fan_out",
    nonlinearity: str = "relu",
    bias: float = 0,
    distribution: str = "normal",
) -> Tensor:
    """Initialize the weights and biases of a module using the Kaiming initialization method.

    Copied from mmengine.model.weight_init.kaiming_init

    Args:
        module (nn.Module): The module to initialize.
        a (float): The negative slope of the rectifier used after this layer (only used with 'leaky_relu' nonlinearity).
            Default is 0.
        mode (str): Either 'fan_in' (default) or 'fan_out'. Choosing 'fan_in' preserves the magnitude of the variance
            of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes in the backward pass.
            Default is 'fan_out'.
        nonlinearity (str): The non-linear function (nn.functional name), recommended to use 'relu' or 'leaky_relu'.
            Default is 'relu'.
        bias (float): The bias value. Default is 0.
        distribution (str): The type of distribution to use for weight initialization,
            either 'normal' (default) or 'uniform'.

    Returns:
        Tensor: The initialized tensor.
    """
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


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


def _no_grad_trunc_normal_(tensor: Tensor, mean: float, std: float, a: float, b: float) -> Tensor:
    # Method based on
    # https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    # Modified from
    # https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py
    def norm_cdf(x: float) -> float:
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [lower, upper], then translate
        # to [2lower-1, 2upper-1].
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0) -> Tensor:
    r"""Fills the input Tensor with values drawn from a truncated normal distribution.

    The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Modified from
    https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py

    Args:
        tensor (``torch.Tensor``): an n-dimensional `torch.Tensor`.
        mean (float): the mean of the normal distribution.
        std (float): the standard deviation of the normal distribution.
        a (float): the minimum cutoff value.
        b (float): the maximum cutoff value.
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
