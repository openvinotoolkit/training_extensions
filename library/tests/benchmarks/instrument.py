# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Non-invasive instrumentation of the instance-segmentation data path."""

from __future__ import annotations

import functools
import importlib
from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, Callable, Iterator, ParamSpec, TypeVar
from unittest.mock import patch

from getitune.data.augmentation import pipeline as pipeline_mod
from getitune.data.dataset import base as dataset_base
from getitune.data.entity.sample import SampleBatch

if TYPE_CHECKING:
    from tests.benchmarks.timing import StageTimer

# ``mask/__init__.py`` re-exports the *function* ``mask_target``, which shadows the
# submodule of the same name — import it explicitly to get the module object.
mask_target_mod = importlib.import_module("getitune.data.utils.structures.mask.mask_target")


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _wrap(fn: Callable[_P, _R], timer: StageTimer, name: str, *, sync: bool = False) -> Callable[_P, _R]:
    @functools.wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with timer.record(name, sync=sync):
            return fn(*args, **kwargs)

    return wrapper


@contextmanager
def instrument_pipeline(timer: StageTimer) -> Iterator[StageTimer]:
    """Install timing hooks on stages 2, 4, 5 and 7 of the roadmap table.

    Stage 1 (``convert_to_schema``) is timed by :func:`time_schema_conversion`
    because it runs once at dataset construction, not per sample. Stage 3
    (``Resize``) is handled by :func:`instrument_cpu_transforms`, which needs
    the live pipeline instance to know which transform classes to patch.
    """
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                dataset_base.VisionDataset,
                "_read_dm_item",
                _wrap(dataset_base.VisionDataset._read_dm_item, timer, "read_dm_item"),
            )
        )
        stack.enter_context(
            patch.object(
                dataset_base,
                "_default_collate_fn",
                _wrap(dataset_base._default_collate_fn, timer, "collate"),
            )
        )
        stack.enter_context(
            patch.object(
                SampleBatch,
                "pin_memory",
                _wrap(SampleBatch.pin_memory, timer, "pin_memory"),
            )
        )
        # Module-global lookup at call time (multi_apply resolves it inside the
        # function body), so patching the module attribute takes effect.
        stack.enter_context(
            patch.object(
                mask_target_mod,
                "mask_target_single",
                _wrap(mask_target_mod.mask_target_single, timer, "mask_target_single", sync=True),
            )
        )
        yield timer


@contextmanager
def instrument_cpu_transforms(cpu_pipeline: object, timer: StageTimer) -> Iterator[None]:
    """Time each CPU transform in ``cpu_pipeline.augmentations`` by class name.

    Patches the transform *classes* rather than wrapping the instances:
    ``CPUAugmentationPipeline.forward`` calls ``_is_native_torchvision_transform``
    on each entry, and a proxy object would fail that check and silently reroute
    the transform down the non-native branch — changing what you are measuring.
    """
    augmentations = getattr(cpu_pipeline, "augmentations", None) or []
    classes = {type(t) for t in augmentations}
    with ExitStack() as stack:
        for cls in classes:
            if not hasattr(cls, "forward"):
                continue
            stack.enter_context(patch.object(cls, "forward", _wrap(cls.forward, timer, f"transform:{cls.__name__}")))
        stack.enter_context(
            patch.object(
                pipeline_mod.CPUAugmentationPipeline,
                "forward",
                _wrap(pipeline_mod.CPUAugmentationPipeline.forward, timer, "cpu_pipeline_total"),
            )
        )
        yield


@contextmanager
def time_schema_conversion(timer: StageTimer) -> Iterator[None]:
    """Time ``Dataset.convert_to_schema`` (roadmap stage 1).

    This is the roadmap's flagged re-ranking fork. If this stage dominates,
    ``convert_to_schema`` is eager — it rasterizes the whole subset up front,
    Phase 2a alone recovers most of the win, and Phases 4-6 drop in priority.
    Check ``eager_ratio`` in the emitted report before writing any Phase 1 code.
    """
    from datumaro.experimental import Dataset as DmDataset

    with patch.object(
        DmDataset,
        "convert_to_schema",
        _wrap(DmDataset.convert_to_schema, timer, "convert_to_schema"),
    ):
        yield
