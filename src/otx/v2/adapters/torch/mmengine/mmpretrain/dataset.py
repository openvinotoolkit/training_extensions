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
from otx.v2.adapters.torch.mmengine.modules.utils import CustomConfig as Config
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
        config: Optional[Union[str, Dict[str, Any], Config]] = None,
    ) -> Optional[TorchDataset]:
        if not self.initialize:
            self._initialize()

        if subset not in SUBSET_LIST:
            raise ValueError(f"{subset} is not supported subset")

        # Config Setting
        if isinstance(config, str):
            config = Config.fromfile(filename=config)
        elif isinstance(config, dict):
            config = Config(cfg_dict=config)

        otx_dataset = self.dataset_entity.get_subset(str_to_subset_type(subset))
        labels = self.label_schema.get_labels(include_empty=False)
        if len(otx_dataset) < 1:
            return None
        # Case without config
        if config is None:
            pipeline = pipeline if pipeline is not None else get_default_pipeline()
            dataset = self.base_dataset(otx_dataset=otx_dataset, labels=labels, pipeline=pipeline)
            dataset._build_config = {
                "type": str(self.base_dataset.__qualname__),
                "data_root": getattr(self, f"{subset}_data_roots"),
                "ann_file": getattr(self, f"{subset}_ann_files"),
                "data_prefix": "",
                "pipeline": pipeline,
            }
            return dataset

        # Case with Config
        dataset_config = config.get("dataset", None)
        if dataset_config is None:
            raise ValueError("The config used does not have a dataset.")
        init_config = dataset_config.copy()
        dataset_config["otx_dataset"] = otx_dataset
        dataset_config["labels"] = labels
        if pipeline is not None:
            dataset_config["pipeline"] = pipeline
        dataset_config.pop("data_roots", None)
        dataset_config.pop("ann_files", None)
        dataset_config.pop("file_list", None)
        dataset = mmpretrain_build_dataset(dataset_config)
        dataset._build_config = init_config
        return dataset

    def build_dataloader(
        self,
        dataset: Optional[TorchDataset],
        batch_size: int = 1,
        num_workers: int = 0,
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

        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed) if seed is not None else None
        if digit_version(torch.__version__) >= digit_version("1.8.0"):
            kwargs["persistent_workers"] = persistent_workers

        dataloader = TorchDataLoader(
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
        sampler_cfg = sampler if isinstance(sampler, dict) else dict(type=f"{sampler.__class__.__qualname__}")
        dataset_cfg = dataset._build_config if hasattr(dataset, "_build_config") else dataset
        dataloader._build_config = dict(
            batch_size=batch_size,
            sampler=sampler_cfg,
            num_workers=num_workers,
            collate_fn=dict(type="default_collate"),
            pin_memory=pin_memory,
            shuffle=shuffle,
            dataset=dataset_cfg,
        )
        return dataloader

    def subset_dataloader(
        self,
        subset: str,
        pipeline: Optional[Union[List[Union[Dict, Any]], Dict[str, List[Union[Dict, Any]]]]] = None,
        config: Optional[Union[str, Dict[str, Any]]] = None,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
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
            config = Config.fromfile(filename=config)
        elif isinstance(config, dict):
            config = Config(cfg_dict=config)
        elif config is None:
            config = Config(cfg_dict={})
        dataloader_config = config.get(f"{subset}_dataloader", None)
        subset_pipeline = pipeline
        if isinstance(subset_pipeline, dict):
            subset_pipeline = subset_pipeline[subset]
        subset_dataset = self.build_dataset(subset=subset, pipeline=pipeline, config=dataloader_config)
        # TODO: argument update with configuration (config is not None case)
        if batch_size is None:
            batch_size = config.get("batch_size", 1)
        if num_workers is None:
            num_workers = config.get("num_workers", 0)

        subset_dataloader = self.build_dataloader(
            dataset=subset_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
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
