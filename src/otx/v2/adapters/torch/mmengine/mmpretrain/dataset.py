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
        r"""MMPretrain's Dataset class.

        Args:
            task (Optional[Union[TaskType, str]], optional): The task type of the dataset want to load. Defaults to None.
            train_type (Optional[Union[TrainType, str]], optional): The train type of the dataset want to load. Defaults to None.
            train_data_roots (Optional[str], optional): The root address of the dataset to be used for training. Defaults to None.
            train_ann_files (Optional[str], optional): Location of the annotation file for the dataset to be used for training. Defaults to None.
            val_data_roots (Optional[str], optional): The root address of the dataset to be used for validation. Defaults to None.
            val_ann_files (Optional[str], optional): Location of the annotation file for the dataset to be used for validation. Defaults to None.
            test_data_roots (Optional[str], optional): The root address of the dataset to be used for testing. Defaults to None.
            test_ann_files (Optional[str], optional): Location of the annotation file for the dataset to be used for testing. Defaults to None.
            unlabeled_data_roots (Optional[str], optional): The root address of the unlabeled dataset to be used for training. Defaults to None.
            unlabeled_file_list (Optional[str], optional): The file where the list of unlabeled images is declared. Defaults to None.
            data_format (Optional[str], optional): The format of the dataset. Defaults to None.
        """
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
        dataset_config["_scope_"] = "mmpretrain"
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
    ) -> TorchDataLoader:
        r"""MMPretrain's Dataset.subset_dataloader.

        Args:
            subset (str): Enter an available subset of that dataset.
            pipeline (Optional[Union[List[Union[Dict, Any]], Dict[str, List[Union[Dict, Any]]]]], optional): Dataset Pipeline. Defaults to None.
            config (Optional[Union[str, Dict[str, Any]]], optional): Path to configuration file or Config. Defaults to None.
            batch_size (Optional[int], optional): How many samples per batch to load. Defaults to None.
            num_workers (Optional[int], optional): How many subprocesses to use for data loading.
                ``0`` means that the data will be loaded in the main process. Defaults to None.
            shuffle (bool, optional): Set to ``True`` to have the data reshuffled at every epoch. Defaults to True.
            pin_memory (bool, optional): If ``True``, the data loader will copy Tensors
                into device/CUDA pinned memory before returning them.  If your data elements
                are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
                see the example below. Defaults to False.
            drop_last (bool, optional): _description_. Defaults to False.
            sampler (Optional[Union[Sampler, Iterable, Dict]], optional): Defines the strategy to draw
                samples from the dataset. Can be any ``Iterable`` with ``__len__``
                implemented. If specified, :attr:`shuffle` must not be specified.. Defaults to None.
            persistent_workers (bool, optional): If ``True``, the data loader will not shutdown
                the worker processes after a dataset has been consumed once. This allows to
                maintain the workers `Dataset` instances alive. Defaults to False.
            distributed (bool, optional): _description_. Defaults to False.

        Returns:
            torch.utils.data.DataLoader: Returns a subset of dataLoader.
        """
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

        # kwargs conflict
        unlabeled_batch_size = kwargs.pop("unlabeled_batch_size", config.get("unlabeled_batch_size", batch_size))
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
        if not self.initialize:
            self._initialize()
        return len(self.label_schema.get_labels(include_empty=False))


if __name__ == "__main__":
    dataset = Dataset(
        train_data_roots="/home/harimkan/workspace/datasets/otx_cls_dataset/otx_dataset/svhn@4_0/train_data",
        val_data_roots="/home/harimkan/workspace/datasets/otx_cls_dataset/otx_dataset/svhn@4_0/val_data",
        test_data_roots="/home/harimkan/workspace/datasets/otx_cls_dataset/otx_dataset/svhn@4_0/val_data",
    )
    dataloader = dataset.train_dataloader()
