"""Memory cache handler implementations and singleton class to call them."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import ctypes as ct
import multiprocessing as mp
from multiprocessing.managers import DictProxy
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import psutil
from multiprocess.synchronize import Lock

from otx.utils.logger import get_logger

logger = get_logger()
GIB = 1024**3


class _DummyLock:
    def __enter__(self, *args, **kwargs):
        pass

    def __exit__(self, *args, **kwargs):
        pass


class MemCacheHandlerBase:
    """Base class for memory cache handler.

    It will be combined with LoadImageFromOTXDataset to store/retrieve the samples in memory.
    """

    def __init__(self, mem_size: int):
        self._init_data_structs(mem_size)

    def _init_data_structs(self, mem_size: int):
        self._arr = (ct.c_uint8 * mem_size)()
        self._cur_page = ct.c_size_t(0)
        self._cache_addr: Union[Dict, DictProxy] = {}
        self._lock: Union[Lock, _DummyLock] = _DummyLock()
        self._freeze = ct.c_bool(False)

    def __len__(self):
        """Get the number of cached items."""
        return len(self._cache_addr)

    @property
    def mem_size(self) -> int:
        """Get the reserved memory pool size (bytes)."""
        return len(self._arr)

    def get(self, key: Any) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Try to look up the cached item with the given key.

        Args:
            key (Any): A key for looking up the cached item

        Returns:
            If succeed return (np.ndarray, Dict), otherwise return (None, None)
        """
        if self.mem_size == 0 or key not in self._cache_addr:
            return None, None

        addr = self._cache_addr[key]

        offset, count, dtype, shape, strides, meta = addr

        data = np.frombuffer(self._arr, dtype=dtype, count=count, offset=offset)
        return np.lib.stride_tricks.as_strided(data, shape, strides), meta

    def put(self, key: Any, data: np.ndarray, meta: Optional[Dict] = None) -> Optional[int]:
        """Try to store np.ndarray and metadata with a key to the reserved memory pool.

        Args:
            key (Any): A key to store the cached item
            data (np.ndarray): A data sample to store
            meta (Optional[Dict]): A metadata of the data sample

        Returns:
            Optional[int]: If succeed return the address of cached item in memory pool
        """
        if self._freeze.value:
            return None

        data_bytes = data.size * data.itemsize

        with self._lock:
            new_page = self._cur_page.value + data_bytes

            if key in self._cache_addr or new_page > self.mem_size:
                return None

            offset = ct.byref(self._arr, self._cur_page.value)
            ct.memmove(offset, data.ctypes.data, data_bytes)

            self._cache_addr[key] = (
                self._cur_page.value,
                data.size,
                data.dtype,
                data.shape,
                data.strides,
                meta,
            )
            self._cur_page.value = new_page
            return new_page

    def __repr__(self):
        """Representation for the current handler status."""
        perc = 100.0 * self._cur_page.value / self.mem_size if self.mem_size > 0 else 0.0
        return (
            f"{self.__class__.__name__} "
            f"uses {self._cur_page.value} / {self.mem_size} ({perc:.1f}%) memory pool and "
            f"store {len(self)} items."
        )

    def freeze(self):
        """If frozen, it is impossible to store a new item anymore."""
        self._freeze.value = True

    def unfreeze(self):
        """If unfrozen, it is possible to store a new item."""
        self._freeze.value = False


class MemCacheHandlerForSP(MemCacheHandlerBase):
    """Memory caching handler for single processing.

    Use if PyTorch's DataLoader.num_workers == 0.
    """


class MemCacheHandlerForMP(MemCacheHandlerBase):
    """Memory caching handler for multi processing.

    Use if PyTorch's DataLoader.num_workers > 0.
    """

    def _init_data_structs(self, mem_size: int):
        self._arr = mp.Array(ct.c_uint8, mem_size, lock=False)
        self._cur_page = mp.Value(ct.c_size_t, 0, lock=False)

        self._manager = mp.Manager()
        self._cache_addr: DictProxy = self._manager.dict()
        self._lock = mp.Lock()
        self._freeze = mp.Value(ct.c_bool, False, lock=False)

    def __del__(self):
        """When deleting, manager should also be shutdowned."""
        self._manager.shutdown()


class MemCacheHandlerError(Exception):
    """Exception class for MemCacheHandler."""


class MemCacheHandlerSingleton:
    """A singleton class to create, delete and get MemCacheHandlerBase."""

    instance: MemCacheHandlerBase
    CPU_MEM_LIMITS_GIB: int = 30

    @classmethod
    def get(cls) -> MemCacheHandlerBase:
        """Get the created MemCacheHandlerBase.

        If no one is created before, raise RuntimeError.
        """
        if not hasattr(cls, "instance"):
            cls_name = cls.__class__.__name__
            raise MemCacheHandlerError(f"Before calling {cls_name}.get(), you should call {cls_name}.create() first.")

        return cls.instance

    @classmethod
    def create(cls, mode: str, mem_size: int) -> MemCacheHandlerBase:
        """Create a new MemCacheHandlerBase instance.

        Args:
            mode (str): There are two options: null, multiprocessing or singleprocessing.
            mem_size (int): The size of memory pool (bytes).
        """

        # COPY FROM mmcv.runner.get_dist_info
        from torch import distributed

        if distributed.is_available() and distributed.is_initialized():
            world_size = distributed.get_world_size()
        else:
            world_size = 1

        # Prevent CPU OOM issue
        memory_info = psutil.virtual_memory()
        available_cpu_mem = memory_info.available / GIB

        if world_size > 1:
            mem_size = mem_size // world_size
            available_cpu_mem = available_cpu_mem // world_size
            logger.info(f"Since world_size={world_size} > 1, each worker a {mem_size} size memory pool.")

        logger.info(f"Try to create a {mem_size} size memory pool.")
        if available_cpu_mem < ((mem_size / GIB) + cls.CPU_MEM_LIMITS_GIB):
            logger.warning("No available CPU memory left, mem_size will be set to 0.")
            mem_size = 0

        if mode == "null" or mem_size == 0:
            cls.instance = MemCacheHandlerBase(mem_size=0)
            cls.instance.freeze()
        elif mode == "multiprocessing":
            cls.instance = MemCacheHandlerForMP(mem_size)
        elif mode == "singleprocessing":
            cls.instance = MemCacheHandlerForSP(mem_size)
        else:
            raise MemCacheHandlerError(f"{mode} is unknown mode.")

        return cls.instance

    @classmethod
    def delete(cls) -> None:
        """Delete the existing MemCacheHandlerBase instance."""
        if hasattr(cls, "instance"):
            del cls.instance
