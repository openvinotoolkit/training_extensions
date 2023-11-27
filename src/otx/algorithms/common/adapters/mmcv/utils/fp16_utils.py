"""Custom fp16 related modules to enable XPU modules."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import functools
from inspect import getfullargspec
from typing import Callable, Iterable, Optional

import torch
from mmcv.runner.fp16_utils import cast_tensor_type
from mmcv.utils import IS_NPU_AVAILABLE, TORCH_VERSION, digit_version
from torch import nn

from otx.algorithms.common.utils import is_xpu_available

try:
    if is_xpu_available():
        from torch.xpu.amp import autocast
    elif IS_NPU_AVAILABLE:
        from torch.npu.amp import autocast
    else:
        from torch.cuda.amp import autocast
except ImportError:
    pass


def custom_auto_fp16(
    apply_to: Optional[Iterable] = None,
    out_fp32: bool = False,
    supported_types: tuple = (nn.Module,),
) -> Callable:
    """Custom decorator to enable fp16 training automatically on XPU as well."""

    def auto_fp16_wrapper(old_func: Callable) -> Callable:
        @functools.wraps(old_func)
        def new_func(*args, **kwargs) -> Callable:
            # check if the module has set the attribute `fp16_enabled`, if not,
            # just fallback to the original method.
            if not isinstance(args[0], supported_types):
                raise TypeError(
                    "@auto_fp16 can only be used to decorate the " f"method of those classes {supported_types}"
                )
            if not (hasattr(args[0], "fp16_enabled") and args[0].fp16_enabled):
                return old_func(*args, **kwargs)

            target_dtype = torch.bfloat16 if is_xpu_available() else torch.half
            # get the arg spec of the decorated method
            args_info = getfullargspec(old_func)
            # get the argument names to be casted
            args_to_cast = args_info.args if apply_to is None else apply_to
            # convert the args that need to be processed
            new_args = []
            # NOTE: default args are not taken into consideration
            if args:
                arg_names = args_info.args[: len(args)]
                for i, arg_name in enumerate(arg_names):
                    if arg_name in args_to_cast:
                        new_args.append(cast_tensor_type(args[i], torch.float, target_dtype))
                    else:
                        new_args.append(args[i])
            # convert the kwargs that need to be processed
            new_kwargs = {}
            if kwargs:
                for arg_name, arg_value in kwargs.items():
                    if arg_name in args_to_cast:
                        new_kwargs[arg_name] = cast_tensor_type(arg_value, torch.float, target_dtype)
                    else:
                        new_kwargs[arg_name] = arg_value
            # apply converted arguments to the decorated method
            if TORCH_VERSION != "parrots" and digit_version(TORCH_VERSION) >= digit_version("1.6.0"):
                with autocast(enabled=True):
                    output = old_func(*new_args, **new_kwargs)
            else:
                output = old_func(*new_args, **new_kwargs)
            # cast the results back to fp32 if necessary
            if out_fp32:
                output = cast_tensor_type(output, target_dtype, torch.float)
            return output

        return new_func

    return auto_fp16_wrapper


def custom_force_fp32(apply_to: Optional[Iterable] = None, out_fp16: bool = False) -> Callable:
    """Custom decorator to convert input arguments to fp32 in force on XPU as well."""

    def force_fp32_wrapper(old_func):
        @functools.wraps(old_func)
        def new_func(*args, **kwargs) -> Callable:
            # check if the module has set the attribute `fp16_enabled`, if not,
            # just fallback to the original method.
            if not isinstance(args[0], torch.nn.Module):
                raise TypeError("@force_fp32 can only be used to decorate the " "method of nn.Module")
            if not (hasattr(args[0], "fp16_enabled") and args[0].fp16_enabled):
                return old_func(*args, **kwargs)

            source_dtype = torch.bfloat16 if is_xpu_available() else torch.half
            # get the arg spec of the decorated method
            args_info = getfullargspec(old_func)
            # get the argument names to be casted
            args_to_cast = args_info.args if apply_to is None else apply_to
            # convert the args that need to be processed
            new_args = []
            if args:
                arg_names = args_info.args[: len(args)]
                for i, arg_name in enumerate(arg_names):
                    if arg_name in args_to_cast:
                        new_args.append(cast_tensor_type(args[i], source_dtype, torch.float))
                    else:
                        new_args.append(args[i])
            # convert the kwargs that need to be processed
            new_kwargs = dict()
            if kwargs:
                for arg_name, arg_value in kwargs.items():
                    if arg_name in args_to_cast:
                        new_kwargs[arg_name] = cast_tensor_type(arg_value, source_dtype, torch.float)
                    else:
                        new_kwargs[arg_name] = arg_value
            # apply converted arguments to the decorated method
            if TORCH_VERSION != "parrots" and digit_version(TORCH_VERSION) >= digit_version("1.6.0"):
                with autocast(enabled=False):
                    output = old_func(*new_args, **new_kwargs)
            else:
                output = old_func(*new_args, **new_kwargs)
            # cast the results back to fp32 if necessary
            if out_fp16:
                output = cast_tensor_type(output, torch.float, source_dtype)
            return output

        return new_func

    return force_fp32_wrapper
