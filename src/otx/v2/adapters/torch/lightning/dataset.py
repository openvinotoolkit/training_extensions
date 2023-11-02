"""OTX adapters.torch.lightning.Dataset API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from omegaconf import DictConfig, OmegaConf

from otx.v2.adapters.torch.dataset import BaseTorchDataset
from otx.v2.adapters.torch.lightning.modules.datasets import OTXVisualPromptingDataset
from otx.v2.adapters.torch.lightning.modules.datasets.pipelines import collate_fn
from otx.v2.api.entities.task_type import TaskType
from otx.v2.api.utils.decorators import add_subset_dataloader
from otx.v2.api.utils.type_utils import str_to_subset_type

if TYPE_CHECKING:
    from torch.utils.data import DataLoader as TorchDataLoader
    from torch.utils.data import Dataset as TorchDataset
    from torch.utils.data import Sampler

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
        pipeline: dict | list | None = None,  # transform_config
        config: str | (DictConfig | dict) | None = None,
    ) -> TorchDataset | None:
        """Build a TorchDataset for the given subset using the specified pipeline and configuration.

        Args:
            subset (str): The subset to build the dataset for.
            pipeline (dict | list | None, optional): The pipeline to use for data transformation.
            config (str | (DictConfig | dict) | None, optional): The configuration to use for the dataset.

        Returns:
            TorchDataset | None: The built TorchDataset, or None if the dataset is empty.
        """
        if not self.initialize:
            self._initialize()

        # NOTE: It needs to be refactored in a more general way.
        if self.task in (TaskType.VISUAL_PROMPTING, "visual_prompting"):
            config = OmegaConf.load(filename=config) if isinstance(config, str) else DictConfig({})
            config = config.get("dataset", config)
            image_size = config.get("image_size", 1024)
            normalize = config.get("normalize", {})
            mean = normalize.get("mean", [123.675, 116.28, 103.53])
            std = normalize.get("std", [58.395, 57.12, 57.375])
            offset_bbox = config.get("offset_bbox", 20)

            if subset == "predict":
                otx_dataset = self.dataset_entity
            else:
                otx_dataset = self.dataset_entity.get_subset(str_to_subset_type(subset))
            if len(otx_dataset) < 1:
                return None

            return OTXVisualPromptingDataset(
                dataset=otx_dataset,
                image_size=image_size,
                mean=mean,
                std=std,
                offset_bbox=offset_bbox,
                pipeline=pipeline,
            )
        raise NotImplementedError

    def _build_dataloader(
        self,
        dataset: TorchDataset | None,
        batch_size: int | None = 1,
        num_workers: int | None = 0,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        sampler: Sampler | Iterable | None = None,
        persistent_workers: bool = False,
        **kwargs,
    ) -> TorchDataLoader | None:
        """Build a PyTorch DataLoader object from an TorchDataset object.

        Args:
            dataset (Optional[TorchDataset]): The dataset to load.
            batch_size (Optional[int], optional): The batch size. Defaults to 1.
            num_workers (Optional[int], optional): The number of worker threads to use. Defaults to 0.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
            pin_memory (bool, optional): Whether to pin memory. Defaults to False.
            drop_last (bool, optional): Whether to drop the last batch if it is smaller than the batch size.
                Defaults to False.
            sampler (Optional[Union[Sampler, Iterable]], optional): The sampler to use. Defaults to None.
            persistent_workers (bool, optional): Whether to keep the worker processes alive between data loading
                iterations. Defaults to False.
            **kwargs (Any): Takes additional parameters for the torch Dataloader.

        Returns:
            Optional[TorchDataLoader]: The PyTorch DataLoader object.
        """
        if dataset is None:
            return None
        if sampler is not None:
            shuffle = False

        # Currently, copy from mm's
        input_collate_fn = kwargs.pop("collate_fn", collate_fn)
        return super()._build_dataloader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=input_collate_fn,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            **kwargs,
        )
