from typing import Any, Dict, Optional, Union

import albumentations as A
from omegaconf import DictConfig, OmegaConf
from otx.v2.adapters.torch.anomalib.modules.data.data import OTXAnomalyDataset
from otx.v2.api.core.dataset import BaseDataset
from otx.v2.api.entities.task_type import TaskType, TrainType
from otx.v2.api.utils.dataset_utils import (
    contains_anomalous_images,
    split_local_global_dataset,
)
from otx.v2.api.utils.decorators import add_subset_dataloader
from otx.v2.api.utils.type_utils import str_to_subset_type
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset

from anomalib.data.base.datamodule import collate_fn

SUBSET_LIST = ["train", "val", "test", "predict"]


@add_subset_dataloader(SUBSET_LIST)
class Dataset(BaseDataset):
    def __init__(
        self,
        task: Optional[Union[TaskType, str]] = None,
        train_type: Optional[Union[TrainType, str]] = None,
        train_data_roots: Optional[str] = None,
        train_ann_files: Optional[str] = None,
        val_data_roots: Optional[str] = None,
        val_ann_files: Optional[str] = None,
        test_data_roots: Optional[str] = None,
        test_ann_files: Optional[str] = None,
        unlabeled_data_roots: Optional[str] = None,
        unlabeled_file_list: Optional[str] = None,
        data_format: Optional[str] = "mvtec",
    ) -> None:
        if isinstance(task, str) and not task.startswith("anomaly"):
            task = "anomaly_" + task
        super().__init__(
            task,
            train_type,
            train_data_roots,
            train_ann_files,
            val_data_roots,
            val_ann_files,
            test_data_roots,
            test_ann_files,
            unlabeled_data_roots,
            unlabeled_file_list,
            data_format=data_format,  # TODO: Is there a way to make it more flexible?
        )
        self.initialize = False

    def _initialize(self):
        self.set_datumaro_adapters()
        self.initialize = True

    def build_dataset(
        self,
        subset: str,
        pipeline: Optional[Union[str, A.Compose]] = None,  # transform_config
        config: Optional[Union[str, DictConfig, Dict[str, Any]]] = None,
    ) -> Optional[TorchDataset]:
        if not self.initialize:
            self._initialize()

        if isinstance(config, str):
            config = OmegaConf.load(filename=config)
        else:
            config = DictConfig({})

        config.dataset = {"transform_config": {"train": pipeline}, "image_size": [256, 256]}
        otx_dataset = self.dataset_entity.get_subset(str_to_subset_type(subset))
        if len(otx_dataset) < 1:
            return None

        if subset == "val":
            global_dataset, local_dataset = split_local_global_dataset(otx_dataset)
            if contains_anomalous_images(local_dataset):
                otx_dataset = local_dataset
            else:
                otx_dataset = global_dataset
        return OTXAnomalyDataset(config=config, dataset=otx_dataset, task_type=self.task)

    def build_dataloader(
        self,
        dataset: Optional[OTXAnomalyDataset],
        batch_size: int = 1,
        num_workers: int = 0,
        num_gpus: int = 1,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        sampler=None,
        persistent_workers: bool = False,
        distributed: bool = False,
        **kwargs,
    ) -> Optional[TorchDataLoader]:
        if dataset is None:
            return None
        if sampler is not None:
            shuffle = False

        # TODO: Need to check for anomalib dataloader
        # Currently, copy from mm's
        return TorchDataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            **kwargs,
        )

    def subset_dataloader(
        self,
        subset: str,
        pipeline: Optional[Union[str, A.Compose]] = None,
        config: Optional[Union[str, Dict[str, Any]]] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        num_gpus: int = 1,
        shuffle: bool = True,
        pin_memory: bool = False,
        drop_last: bool = False,
        sampler=None,
        persistent_workers: bool = False,
        distributed: bool = False,
        **kwargs,
    ):
        if subset == "predict":
            pass
        dataset = self.build_dataset(subset=subset, pipeline=pipeline, config=config)
        # TODO: argument update with configuration (config is not None case) + Semi-SL Setting flow
        return self.build_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            num_gpus=num_gpus,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            sampler=sampler,
            persistent_workers=persistent_workers,
            distributed=distributed,
            **kwargs,
        )

    @property
    def num_classes(self):
        if not self.initialize:
            self._initialize()
        return len(self.label_schema.get_labels(include_empty=False))


if __name__ == "__main__":
    dataset = Dataset(
        train_data_roots="/home/harimkan/workspace/repo/otx-fork/tests/assets/anomaly/hazelnut/train",
        val_data_roots="/home/harimkan/workspace/repo/otx-fork/tests/assets/anomaly/hazelnut/test",
        test_data_roots="/home/harimkan/workspace/repo/otx-fork/tests/assets/anomaly/hazelnut/test",
    )
    train_dataloader = dataset.train_dataloader()
