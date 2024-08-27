# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Implementation of diffusion data sample."""

from __future__ import annotations

import numpy as np
from otx.algo.utils.mmengine_utils import InstanceData


class DiffusionDataSample(InstanceData):
    """The Diffusion data structure that is used as the interface between modules.

    The attributes of ``DiffusionDataSample`` include:

        - ``input_ids``(np.ndarray): Tokenized caption input.
    """

    def __init__(self, *, metainfo: dict | None = None, **kwargs) -> None:
        super().__init__(metainfo=metainfo, **kwargs)
        self._input_ids: np.ndarray

    @property
    def input_ids(self) -> np.ndarray:
        """Property of `noise`."""
        return self._input_ids

    @input_ids.setter
    def input_ids(self, noise: np.ndarray) -> None:
        """Setter of `noise`."""
        self.set_field(noise, "_input_ids", dtype=np.ndarray)

    @input_ids.deleter
    def input_ids(self) -> None:
        """Deleter of `noise`."""
        del self._input_ids
