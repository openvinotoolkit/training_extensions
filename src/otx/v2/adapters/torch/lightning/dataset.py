"""OTX adapters.torch.lightning.Dataset API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from otx.v2.adapters.torch.dataset import BaseTorchDataset
from otx.v2.api.utils.decorators import add_subset_dataloader

if TYPE_CHECKING:
    import albumentations as al
    from omegaconf import DictConfig
    from torch.utils.data import Dataset as TorchDataset

SUBSET_LIST = ["train", "val", "test"]


@add_subset_dataloader(SUBSET_LIST)
class LightningDataset(BaseTorchDataset):
    """A class representing a dataset for Lightning."""

    def _initialize(self) -> None:
        self.set_datumaro_adapters()
        self.initialize = True

    def _build_dataset(
        self,
        subset: str,
        pipeline: str | al.Compose | None = None,  # transform_config
        config: str | (DictConfig | dict) | None = None,
    ) -> TorchDataset | None:
        """Build a TorchDataset for the given subset using the specified pipeline and configuration.

        Args:
            subset (str): The subset to build the dataset for.
            pipeline (Optional[Union[str, al.Compose]]): The pipeline to use for data transformation.
            config (Optional[Union[str, DictConfig, dict]]): The configuration to use for the dataset.

        Returns:
            Optional[TorchDataset]: The built TorchDataset, or None if the dataset is empty.
        """
        raise NotImplementedError
