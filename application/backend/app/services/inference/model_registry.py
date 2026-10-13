# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from uuid import UUID

from app.models.system import DeviceInfo
from app.services.inference.model_loader import LoadedModelHandle


class EntryState(StrEnum):
    LOADING = "LOADING"
    READY = "READY"
    EVICTED = "EVICTED"


@dataclass
class ModelCacheEntry:
    """One model slot in the inference server cache, with the bookkeeping needed to share it safely.

    Attributes:
        project_id: Project the model belongs to.
        model_id: Identifier of the cached model.
        variant_id: Identifier of the cached model variant.
        device: Device the model is (being) loaded on.
        xml_path: Path to the OpenVINO IR `.xml` file used to load this entry.
        state: Lifecycle state of the entry.
        handle: Loaded model handle, available once the state is READY.
        error: Exception raised by the load, if it failed.
        refcount: Number of active leases; an entry is evictable only at zero.
        last_used: Monotonic timestamp of the last lease, used for LRU and TTL eviction.
        ready: Set exactly once by the loading thread, on both success and failure.
        infer_lock: Serializes parameter changes and inference on the underlying model object.
    """

    project_id: UUID
    model_id: UUID
    variant_id: UUID
    device: DeviceInfo
    xml_path: Path
    state: EntryState = EntryState.LOADING
    handle: LoadedModelHandle | None = None
    error: BaseException | None = None
    refcount: int = 0
    last_used: float = field(default_factory=time.monotonic)
    ready: threading.Event = field(default_factory=threading.Event)
    infer_lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def key(self) -> tuple[UUID, UUID]:
        """Cache key of this entry: the `(model_id, variant_id)` pair it is registered under."""
        return self.model_id, self.variant_id
