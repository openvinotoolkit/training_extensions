"""Collections of Utils for common OTX algorithms."""

# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib
import inspect
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import onnx
import torch
import yaml
from addict import Dict as adict

from otx.utils.logger import get_logger
from otx.utils.utils import add_suffix_to_filename

logger = get_logger()


HPU_AVAILABLE = None
try:
    import habana_frameworks.torch as htorch
except ImportError:
    HPU_AVAILABLE = False
    htorch = None

XPU_AVAILABLE = None
try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    XPU_AVAILABLE = False
    ipex = None


class UncopiableDefaultDict(defaultdict):
    """Defauldict type object to avoid deepcopy."""

    def __deepcopy__(self, memo):
        """Deepcopy."""
        return self


def load_template(path):
    """Loading model template function."""
    with open(path, encoding="UTF-8") as f:
        template = yaml.safe_load(f)
    return template


def get_task_class(path: str):
    """Return Task classes."""
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def get_arg_spec(  # noqa: C901  # pylint: disable=too-many-branches
    fn: Callable,  # pylint: disable=invalid-name
    depth: Optional[int] = None,
) -> Tuple[str, ...]:
    """Get argument spec of function."""

    args = set()

    cls_obj = None
    if inspect.ismethod(fn):
        fn_name = fn.__name__
        cls_obj = fn.__self__
        if not inspect.isclass(cls_obj):
            cls_obj = cls_obj.__class__
    else:
        fn_name = fn.__name__
        names = fn.__qualname__.split(".")
        if len(names) > 1 and names[-1] == fn_name:
            cls_obj = globals()[".".join(names[:-1])]

    if cls_obj:
        for obj in cls_obj.mro():  # type: ignore
            fn_obj = cls_obj.__dict__.get(fn_name, None)
            if fn_obj is not None:
                if isinstance(fn_obj, staticmethod):
                    cls_obj = None
                    break

    if cls_obj is None:
        # function, staticmethod
        spec = inspect.getfullargspec(fn)
        args.update(spec.args)
    else:
        # method, classmethod
        for i, obj in enumerate(cls_obj.mro()):  # type: ignore
            if depth is not None and i == depth:
                break
            method = getattr(obj, fn_name, None)
            if method is None:
                break
            spec = inspect.getfullargspec(method)
            args.update(spec.args[1:])
            if spec.varkw is None and spec.varargs is None:
                break
    return tuple(args)


def set_random_seed(seed, logger=None, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        logger (logging.Logger): logger for logging seed info
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if is_xpu_available():
        torch.xpu.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if logger:
        logger.info(f"Training seed was set to {seed} w/ deterministic={deterministic}.")
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_default_async_reqs_num() -> int:
    """Returns a default number of infer request for OV models."""
    reqs_num = os.cpu_count()
    if reqs_num is not None:
        reqs_num = max(1, int(reqs_num / 2))
        return reqs_num
    else:
        return 1


def read_py_config(filename: str) -> adict:
    """Reads py config to a dict."""
    filename = str(Path(filename).resolve())
    if not Path(filename).is_file:
        raise RuntimeError("config not found")
    assert filename.endswith(".py")
    module_name = Path(filename).stem
    if "." in module_name:
        raise ValueError("Dots are not allowed in config file path.")
    config_dir = Path(filename).parent
    sys.path.insert(0, str(config_dir))
    mod = importlib.import_module(module_name)
    sys.path.pop(0)
    cfg_dict = adict(
        {
            name: value
            for name, value in mod.__dict__.items()
            if not name.startswith("__") and not inspect.isclass(value) and not inspect.ismodule(value)
        }
    )

    return cfg_dict


def embed_onnx_model_data(onnx_file: str, extra_model_data: Dict[Tuple[str, str], Any]) -> None:
    """Embeds model api config to onnx file."""
    model = onnx.load(onnx_file)

    for item in extra_model_data:
        meta = model.metadata_props.add()
        attr_path = " ".join(map(str, item))
        meta.key = attr_path.strip()
        meta.value = str(extra_model_data[item])

    onnx.save(model, onnx_file)


def is_xpu_available() -> bool:
    """Checks if XPU device is available."""
    global XPU_AVAILABLE  # noqa: PLW0603
    if XPU_AVAILABLE is None:
        XPU_AVAILABLE = hasattr(torch, "xpu") and torch.xpu.is_available()
    return XPU_AVAILABLE


def is_hpu_available() -> bool:
    """Check if HPU device is available."""
    global HPU_AVAILABLE  # noqa: PLW0603
    if HPU_AVAILABLE is None:
        HPU_AVAILABLE = htorch.hpu.is_available()
    return HPU_AVAILABLE


def cast_bf16_to_fp32(tensor: torch.Tensor) -> torch.Tensor:
    """Cast bf16 tensor to fp32 before processed by numpy.

    numpy doesn't support bfloat16, it is required to convert bfloat16 tensor to float32.
    """
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float32)
    return tensor


def get_cfg_based_on_device(cfg_file_path: Union[str, Path]) -> str:
    """Find a config file according to device."""
    if is_xpu_available():
        cfg_for_device = add_suffix_to_filename(cfg_file_path, "_xpu")
        if cfg_for_device.exists():
            logger.info(
                f"XPU is detected. XPU config file will be used : {Path(cfg_file_path).name} -> {cfg_for_device.name}"
            )
            cfg_file_path = cfg_for_device

    return str(cfg_file_path)
