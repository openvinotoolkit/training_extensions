# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for the data module."""

from __future__ import annotations

import importlib
import inspect
import logging
from multiprocessing import cpu_count
from typing import TYPE_CHECKING, Any

import polars as pl
import torch

from getitune.utils.device import is_xpu_available

if TYPE_CHECKING:
    from datumaro.experimental import Dataset
    from torch.utils.data import Dataset as TorchDataset
    from torch.utils.data import Sampler

    from getitune.config.data import SamplerConfig


logger = logging.getLogger(__name__)


def instantiate_sampler(sampler_config: SamplerConfig, dataset: TorchDataset, **kwargs) -> Sampler:
    """Instantiate a sampler object based on the provided configuration.

    Args:
        sampler_config (SamplerConfig): The configuration object for the sampler.
        dataset (Dataset): The dataset object to be sampled.
        **kwargs: Additional keyword arguments to be passed to the sampler's constructor.

    Returns:
        Sampler: The instantiated sampler object.
    """
    class_module, class_name = sampler_config.class_path.rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    sampler_class = getattr(module, class_name)
    init_signature = list(inspect.signature(sampler_class.__init__).parameters.keys())
    if "batch_size" not in init_signature:
        kwargs.pop("batch_size", None)
    # Handle None init_args
    init_args = sampler_config.init_args or {}
    sampler_kwargs = {**init_args, **kwargs}
    return sampler_class(dataset, **sampler_kwargs)


def get_adaptive_num_workers(num_dataloader: int = 1) -> int | None:
    """Measure appropriate num_workers value and return it."""
    num_devices = torch.xpu.device_count() if is_xpu_available() else torch.cuda.device_count()
    if num_devices == 0:
        return None
    return min(cpu_count() // (num_dataloader * num_devices), 8)  # max available num_workers is 8


def import_object_from_module(obj_path: str) -> Any:  # noqa: ANN401
    """Get object from import format string."""
    module_name, obj_name = obj_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


def fill_null_annotation_lists(dm_dataset: Dataset) -> Dataset:
    """Replace ``None`` list-typed annotation columns with empty lists.

    getitune datasets legitimately store ``None`` (rather than an empty list) for
    list-typed annotation fields (e.g. ``labels``, ``bboxes``, ``polygons``) on
    samples with no annotations at all (e.g. background-only images). Datumaro's
    tiling transforms (``LabelTiler``, ``PolygonTiler``, ...) iterate over these
    columns with plain Python loops and don't guard against ``None``, which
    raises ``TypeError: object of type 'NoneType' has no len()`` (or similar)
    when such a sample is tiled. Sanitize the dataframe up front so tiling never
    sees a ``None`` where an (empty) list is expected.

    Only simple scalar ``List`` columns (e.g. ``List(UInt32)`` for labels) are
    filled. Columns whose (possibly nested) inner type is a fixed-size ``Array``
    or ``Struct`` (e.g. ``List(Array(Float32, shape=(4,)))`` for bboxes, or
    ``List(List(Array(Float32, shape=(2,))))`` for polygons) are left unchanged
    because ``fill_null([])`` can corrupt the dtype and because the corresponding
    datumaro tilers (``BboxTiler``, …) already handle ``None`` gracefully via
    Polars expressions.
    """
    try:
        null_list_columns = [
            name
            for name, dtype in dm_dataset.df.schema.items()
            if isinstance(dtype, pl.List) and _is_simple_scalar_list(dtype)
        ]
    except (AttributeError, TypeError):
        # Defensive: don't let sanitization itself break dataset construction for
        # dataset-like objects (e.g. test doubles) that don't expose a real
        # datumaro-backed dataframe.
        return dm_dataset
    if null_list_columns:
        dm_dataset.df = dm_dataset.df.with_columns([pl.col(name).fill_null([]) for name in null_list_columns])
    return dm_dataset


def _is_simple_scalar_list(dtype: pl.List) -> bool:
    """Return True if a ``polars.List`` column holds a simple scalar inner type.

    For ``List(Array(…))`` or ``List(Struct(…))`` columns — including nested cases
    such as ``List(List(Array(…)))`` (e.g. ``polygons``) — ``fill_null([])``
    produces an empty list whose dtype does not round-trip properly through
    Datumaro's schema conversion (empty ``(0,)`` becomes ``(1,0)``). Unwrap any
    level of list nesting before checking the innermost element type.
    """
    inner = dtype.inner
    while isinstance(inner, pl.List):
        inner = inner.inner
    return not isinstance(inner, (pl.Array, pl.Struct))
