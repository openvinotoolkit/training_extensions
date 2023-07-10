from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Union

import torch
from mmpretrain.datasets import (
    build_dataset as mmpretrain_build_dataset,
)
from otx.v2.adapters.torch.mmengine.mmpretrain.modules.datasets import (
    OTXClsDataset,
    OTXHierarchicalClsDataset,
    OTXMultilabelClsDataset,
    SelfSLDataset,
)
from otx.v2.adapters.torch.mmengine.modules.utils import CustomConfig
from otx.v2.adapters.torch.modules.dataloaders import ComposedDL
from otx.v2.api.core.dataset import BaseDataset
from otx.v2.api.entities.task_type import TaskType, TrainType
from otx.v2.api.utils.decorators import add_subset_dataloader
from otx.v2.api.utils.type_utils import str_to_subset_type
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Sampler

from mmengine.dataset import default_collate, worker_init_fn
from mmengine.dist import get_dist_info
from mmengine.utils import digit_version

SUBSET_LIST = ["train", "val", "test", "unlabeled"]


def get_default_pipeline():
    # TODO: This is function for experiment // Need to remove this function
    try:
        import mmpretrain

        return [
            dict(type="Resize", scale=[224, 224]),
            dict(type="mmpretrain.PackInputs"),
        ]
    except:
        return [
            dict(type="Resize", size=224),
            dict(type="PILImageToNDArray", keys=["img"]),
            dict(type="Normalize", mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="ToTensor", keys=["gt_label"]),
            dict(type="Collect", keys=["img", "gt_label"]),
        ]


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
        data_format: Optional[str] = None,
    ) -> None:
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
            data_format,
        )
        self.initialize = False

    def _initialize(self):
        self.set_datumaro_adapters()  # Set self.dataset_entity & self.label_schema
        self.base_dataset = self._get_sub_task_dataset()
        self.initialize = True

    def _get_sub_task_dataset(self):
        if self.train_type == TrainType.Selfsupervised:
            return SelfSLDataset
        len_group = len(self.label_schema.get_groups(False))
        len_labels = len(self.label_schema.get_labels(include_empty=False))
        if len_group > 1:
            if len_group == len_labels:
                return OTXMultilabelClsDataset
            return OTXHierarchicalClsDataset
        return OTXClsDataset

    def build_dataset(
        self,
        subset: str,
        pipeline: Optional[List[Union[Dict, Any]]] = None,
        config: Optional[Union[str, Dict[str, Any], CustomConfig]] = None,
    ) -> Optional[TorchDataset]:
        if not self.initialize:
            self._initialize()

        if subset not in SUBSET_LIST:
            raise ValueError(f"{subset} is not supported subset")

        # Config Setting
        if isinstance(config, str):
            config = CustomConfig.fromfile(filename=config)
        elif isinstance(config, dict):
            config = CustomConfig(cfg_dict=config)

        otx_dataset = self.dataset_entity.get_subset(str_to_subset_type(subset))
        labels = self.label_schema.get_labels(include_empty=False)
        if len(otx_dataset) < 1:
            return None
        # Case without config
        if config is None:
            pipeline = pipeline if pipeline is not None else get_default_pipeline()
            return self.base_dataset(otx_dataset=otx_dataset, labels=labels, pipeline=pipeline)

        # Case with config
        if subset not in config["data"]:
            raise ValueError(f"{subset} is not in dataset config")
        config = config["data"][subset]
        config["otx_dataset"] = otx_dataset
        config["labels"] = labels
        if pipeline is not None:
            config["pipeline"] = pipeline
        config.pop("data_roots", None)
        config.pop("ann_files", None)
        config.pop("file_list", None)
        return mmpretrain_build_dataset(config)

    def build_dataloader(
        self,
        dataset: Optional[TorchDataset],
        batch_size: int = 1,
        num_workers: int = 0,
        num_gpus: int = 1,
        shuffle: bool = True,
        pin_memory: bool = False,
        drop_last: bool = False,
        sampler: Optional[Union[Sampler, Iterable, Dict]] = None,
        persistent_workers: bool = False,
        distributed: bool = False,
        **kwargs,
    ) -> Optional[TorchDataLoader]:
        if dataset is None:
            return None
        rank, world_size = get_dist_info()
        # TODO: Setting for Semi-SL (training + unlabeled: List[TorchDataset])

        # Sampler
        seed = kwargs.get("seed", None)
        if isinstance(sampler, dict):
            pass
        if sampler is not None:
            shuffle = False

        # if distributed:
        #     samples_per_gpu = batch_size
        # else:
        #     samples_per_gpu = int(batch_size // num_gpus)

        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed) if seed is not None else None
        if digit_version(torch.__version__) >= digit_version("1.8.0"):
            kwargs["persistent_workers"] = persistent_workers

        return TorchDataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(default_collate),
            pin_memory=pin_memory,
            shuffle=shuffle,
            worker_init_fn=init_fn,
            drop_last=drop_last,
            **kwargs,
        )

    def subset_dataloader(
        self,
        subset: str,
        pipeline: Optional[Union[List[Union[Dict, Any]], Dict[str, List[Union[Dict, Any]]]]] = None,
        config: Optional[Union[str, Dict[str, Any]]] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        num_gpus: int = 1,
        shuffle: bool = True,
        pin_memory: bool = False,
        drop_last: bool = False,
        sampler: Optional[Union[Sampler, Iterable, Dict]] = None,
        persistent_workers: bool = False,
        distributed: bool = False,
        **kwargs,
    ):
        # Config Setting
        if isinstance(config, str):
            config = CustomConfig.fromfile(filename=config)
        elif isinstance(config, dict):
            config = CustomConfig(cfg_dict=config)
        subset_pipeline = pipeline
        if isinstance(subset_pipeline, dict):
            subset_pipeline = subset_pipeline[subset]
        subset_dataset = self.build_dataset(subset=subset, pipeline=pipeline, config=config)
        # TODO: argument update with configuration (config is not None case)
        subset_dataloader = self.build_dataloader(
            dataset=subset_dataset,
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
        if subset == "train" and self.train_type == TrainType.Semisupervised:
            unlabeled_pipeline = None
            if pipeline is not None:
                unlabeled_pipeline = pipeline["unlabeled"]
            unlabeled_dataset = self.build_dataset(subset="unlabeled", pipeline=unlabeled_pipeline, config=config)
            unlabeled_batch_size = kwargs.get("unlabeled_batch_size", config.get("unlabeled_batch_size", batch_size))
            unlabeled_dataloader = self.build_dataloader(
                dataset=unlabeled_dataset,
                batch_size=unlabeled_batch_size,
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
            return ComposedDL([subset_dataloader, unlabeled_dataloader])
        return subset_dataloader

    @property
    def num_classes(self):
        return len(self.label_schema.get_labels(include_empty=False))


if __name__ == "__main__":
    dataset = Dataset(
        train_data_roots="/home/harimkan/workspace/datasets/otx_cls_dataset/otx_dataset/svhn@4_0/train_data",
        val_data_roots="/home/harimkan/workspace/datasets/otx_cls_dataset/otx_dataset/svhn@4_0/val_data",
        test_data_roots="/home/harimkan/workspace/datasets/otx_cls_dataset/otx_dataset/svhn@4_0/val_data",
    )
    dataloader = dataset.train_dataloader()
