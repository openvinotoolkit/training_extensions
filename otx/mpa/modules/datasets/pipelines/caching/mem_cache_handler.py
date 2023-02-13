# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import ctypes as ct
import multiprocessing as mp
import re
from typing import Optional

import numpy as np

from otx.mpa.utils.logger import get_logger

logger = get_logger()


class _DummyLock:
    def __enter__(self, *args, **kwargs):
        pass

    def __exit__(self, *args, **kwargs):
        pass


class MemCacheHandlerForSP:
    def __init__(self, mem_size: int):
        self._init_data_structs(mem_size)

    def _init_data_structs(self, mem_size: int):
        self.arr = (ct.c_uint8 * mem_size)()
        self.cur_page = ct.c_size_t(0)
        self.cache_addr = {}
        self.lock = _DummyLock()

    def __len__(self):
        return len(self.cache_addr)

    @property
    def mem_size(self) -> int:
        return len(self.arr)

    def get(self, key: str) -> Optional[np.ndarray]:
        if key not in self.cache_addr:
            return None

        addr = self.cache_addr[key]

        offset, count, shape, strides = addr

        data = np.frombuffer(self.arr, dtype=np.uint8, count=count, offset=offset)
        return np.lib.stride_tricks.as_strided(data, shape, strides)

    def put(self, key: str, data: np.ndarray) -> Optional[int]:
        assert data.dtype == np.uint8

        with self.lock:
            new_page = self.cur_page.value + data.size

            if key in self.cache_addr or new_page > self.mem_size:
                return None

            offset = ct.byref(self.arr, self.cur_page.value)
            ct.memmove(offset, data.ctypes.data, data.size)

            self.cache_addr[key] = (
                self.cur_page.value,
                data.size,
                data.shape,
                data.strides,
            )
            self.cur_page.value = new_page
            return new_page

    def __repr__(self):
        return (
            f"{self.__class__.__name__} "
            f"uses {self.cur_page.value} / {self.mem_size} memory pool and "
            f"store {len(self)} items."
        )


class MemCacheHandlerForMP(MemCacheHandlerForSP):
    def __init__(self, mem_size: int):
        super().__init__(mem_size)

    def _init_data_structs(self, mem_size: int):
        self.arr = mp.Array(ct.c_uint8, mem_size, lock=False)
        self.cur_page = mp.Value(ct.c_size_t, 0, lock=False)

        self.manager = mp.Manager()
        self.cache_addr = self.manager.dict()
        self.lock = mp.Lock()

    def __del__(self):
        self.manager.shutdown()


class MemCacheHandler(MemCacheHandlerForSP):
    instance = Optional[MemCacheHandlerForSP]

    def __init__(self):
        pass

    def __new__(cls) -> Optional[MemCacheHandlerForSP]:
        if not hasattr(cls, "instance"):
            raise RuntimeError(f"Before calling {cls.__name__}(), you should call {cls.__name__}.create() first.")

        return cls.instance

    @classmethod
    def create(cls, mode: str, mem_size: str) -> Optional[MemCacheHandlerForSP]:
        mem_size = cls._parse_mem_size_str(mem_size)
        logger.info(f"Try to create a {mem_size} size memory pool.")

        if mode == "multiprocessing":
            cls.instance = MemCacheHandlerForMP(mem_size)
        elif mode == "singleprocessing":
            cls.instance = MemCacheHandlerForSP(mem_size)
        else:
            raise ValueError(f"{mode} is unknown mode.")

        return cls.instance

    @staticmethod
    def _parse_mem_size_str(mem_size: str) -> int:
        assert isinstance(mem_size, str)

        m = re.match(r"^([\d\.]+)\s*([a-zA-Z]{0,3})$", mem_size.strip())

        if m is None:
            raise ValueError(f"Cannot parse {mem_size} string.")

        units = {
            "": 1,
            "B": 1,
            "KB": 2**10,
            "MB": 2**20,
            "GB": 2**30,
            "KIB": 10**3,
            "MIB": 10**6,
            "GIB": 10**9,
            "K": 2**10,
            "M": 2**20,
            "G": 2**30,
        }

        number, unit = int(m.group(1)), m.group(2).upper()

        if unit not in units:
            raise ValueError(f"{mem_size} has disallowed unit ({unit}).")

        return number * units[unit]
