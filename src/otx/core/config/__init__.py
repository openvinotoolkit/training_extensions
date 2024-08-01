# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Config data type objects."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, _SpecialForm

import yaml

from otx.core.types.task import OTXTaskType, OTXTrainType

if TYPE_CHECKING:
    from torch import dtype


def as_int_tuple(*args) -> tuple[int, ...]:
    """Resolve YAML list into Python integer tuple.

    Example:
        YAML file example::

            ```yaml
            mem_cache_img_max_size: ${as_int_tuple:500,500}
            ```
    """
    return tuple(int(arg) for arg in args)


def as_torch_dtype(arg: str) -> dtype:
    """Resolve YAML string into PyTorch dtype.

    Example:
        YAML file example::

            ```yaml
            transforms:
                - class_path: torchvision.transforms.v2.ToDtype
                  init_args:
                    dtype: ${as_torch_dtype:torch.float32}
                    scale: True
            ```
    """
    import torch

    mapping = {
        "float32": torch.float32,
        "float": torch.float,
        "float64": torch.float64,
        "double": torch.double,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "half": torch.half,
        "uint8": torch.uint8,
        "int8": torch.int8,
        "int16": torch.int16,
        "short": torch.short,
        "int32": torch.int32,
        "int": torch.int,
        "int64": torch.int64,
        "long": torch.long,
        "complex32": torch.complex32,
        "complex64": torch.complex64,
        "cfloat": torch.cfloat,
        "complex128": torch.complex128,
        "cdouble": torch.cdouble,
        "quint8": torch.quint8,
        "qint8": torch.qint8,
        "qint32": torch.qint32,
        "bool": torch.bool,
        "quint4x2": torch.quint4x2,
        "quint2x4": torch.quint2x4,
    }
    prefix = "torch."
    if not arg.startswith(prefix):
        msg = f"arg={arg} should start with the `torch.` prefix"
        raise ValueError(msg)
    key = arg[len(prefix) :]
    return mapping[key]


def dtype_representer(dumper: yaml.Dumper | yaml.representer.SafeRepresenter, data: dtype) -> yaml.ScalarNode:
    """Custom representer for converting dtype object to YAML sequence node.

    Args:
        dumper (yaml.Dumper): The YAML dumper object.
        data (dtype): The dtype object to be converted.

    Returns:
        yaml.Node: The converted YAML node.
    """
    return dumper.represent_str("${as_torch_dtype:" + str(data) + "}")


def any_representer(dumper: yaml.Dumper | yaml.representer.SafeRepresenter, data: Any) -> yaml.ScalarNode:  # noqa: ANN401
    """Representer function that converts any data to a YAML node.

    Args:
        dumper (yaml.Dumper | yaml.representer.SafeRepresenter): The YAML dumper or safe representer.
        data (Any): The data to be represented.

    Returns:
        yaml.Node: The YAML node representing the data.
    """
    return dumper.represent_none(data)


def ignore_aliases(self: yaml.representer.SafeRepresenter, data: Any) -> bool:  # noqa: ARG001, ANN401
    """Determine whether to ignore aliases in YAML representation.

    Args:
        data: The data to check.

    Returns:
        bool | None: True if aliases should be ignored, None otherwise.
    """
    from torch import dtype

    if data is None:
        return True
    if isinstance(data, tuple) and data == ():
        return True
    if isinstance(data, (str, bytes, bool, int, float, dtype)):
        return True
    return None


def otx_str_type_representer(
    dumper: yaml.Dumper | yaml.representer.SafeRepresenter,
    data: OTXTaskType | OTXTrainType,
) -> yaml.ScalarNode:
    """Representer function for converting OTXTaskType or OTXTrainType to a YAML string representation.

    Args:
        dumper (yaml.Dumper | yaml.representer.SafeRepresenter): The YAML dumper or safe representer object.
        data (OTXTaskType | OTXTrainType): The OTXTaskType or OTXTrainType object to be represented.

    Returns:
        yaml.ScalarNode: The YAML ScalarNode representation of the given object.
    """
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(data.value))


def register_configs() -> None:
    """Register custom resolvers."""
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("as_int_tuple", as_int_tuple, replace=True)
    OmegaConf.register_new_resolver("as_torch_dtype", as_torch_dtype, replace=True)

    from torch import dtype

    yaml.add_representer(dtype, dtype_representer)  # For lightnig_logs
    # For jsonargparse's SafeDumper
    yaml.SafeDumper.add_representer(dtype, dtype_representer)
    yaml.SafeDumper.add_representer(_SpecialForm, any_representer)  # typing.Any for DictConfig
    yaml.SafeDumper.ignore_aliases = ignore_aliases  # type: ignore  # noqa: PGH003
    yaml.SafeDumper.add_representer(OTXTaskType, otx_str_type_representer)
    yaml.SafeDumper.add_representer(OTXTrainType, otx_str_type_representer)


register_configs()
