"""OTX Core Engine API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Dict, List, Optional, Union

from otx.v2.api.core.registry import BaseRegistry


class Engine:
    def __init__(self, work_dir: str) -> None:
        self.work_dir = work_dir
        self.registry = BaseRegistry(name="base")

    def train(
        self,
        model=None,
        train_dataloader=None,
        val_dataloader=None,
        optimizer=None,
        max_iters: Optional[int] = None,
        max_epochs: Optional[int] = None,
        distributed: Optional[bool] = None,
        seed: Optional[int] = None,
        deterministic: Optional[bool] = None,
        precision: Optional[str] = None,
        val_interval: Optional[int] = None,
        **kwargs,
    ):
        """Perform training.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    def validate(
        self,
        model=None,
        val_dataloader=None,
        precision: Optional[str] = None,
        **kwargs,
    ):
        """Perform validation.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    def test(
        self,
        model=None,
        test_dataloader=None,
        precision: Optional[str] = None,
        **kwargs,
    ):
        """Perform testing.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    def predict(
        self,
        model=None,
        checkpoint: Optional[Union[str, Path]] = None,
        img=None,
        pipeline: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """Perform prediction.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    def export(self, *args, **kwargs):
        """Perform exporting.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()
