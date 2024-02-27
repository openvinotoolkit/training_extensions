# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""SSD object detector for the OTX detection."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from datumaro.components.annotation import Bbox

from otx.algo.utils.mmconfig import read_mmconfig
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.model.entity.detection import MMDetCompatibleModel
from otx.core.utils.build import build_mm_model, modify_num_classes

if TYPE_CHECKING:
    import torch
    from lightning import Trainer
    from mmdet.models.task_modules.prior_generators.anchor_generator import AnchorGenerator
    from mmengine.registry import Registry
    from omegaconf import DictConfig
    from torch import device, nn

    from otx.core.data.dataset.base import OTXDataset


logger = logging.getLogger()


class SSD(MMDetCompatibleModel):
    """Detecion model class for SSD."""

    def __init__(self, num_classes: int, variant: Literal["mobilenetv2"]) -> None:
        model_name = f"ssd_{variant}"
        config = read_mmconfig(model_name=model_name)
        super().__init__(num_classes=num_classes, config=config)
        self.image_size = (1, 3, 864, 864)
        self.tile_image_size = self.image_size
        self._register_load_state_dict_pre_hook(self._set_anchors_hook)

    def _create_model(self) -> nn.Module:
        from mmdet.models.data_preprocessors import (
            DetDataPreprocessor as _DetDataPreprocessor,
        )
        from mmdet.registry import MODELS
        from mmengine.registry import MODELS as MMENGINE_MODELS

        # NOTE: For the history of this monkey patching, please see
        # https://github.com/openvinotoolkit/training_extensions/issues/2743
        @MMENGINE_MODELS.register_module(force=True)
        class DetDataPreprocessor(_DetDataPreprocessor):
            @property
            def device(self) -> device:
                try:
                    buf = next(self.buffers())
                except StopIteration:
                    return super().device
                else:
                    return buf.device

        self.classification_layers = self.get_classification_layers(self.config, MODELS, "model.")
        return build_mm_model(self.config, MODELS, self.load_from)

    def setup_callback(self, trainer: Trainer) -> None:
        """Callback for setup OTX Model.

        OTXSSD requires auto anchor generating w.r.t. training dataset for better accuracy.
        This callback will provide training dataset to model's anchor generator.

        Args:
            trainer(Trainer): Lightning trainer contains OTXLitModule and OTXDatamodule.
        """
        if trainer.training:
            anchor_generator = self.model.bbox_head.anchor_generator
            dataset = trainer.datamodule.train_dataloader().dataset
            new_anchors = self._get_new_anchors(dataset, anchor_generator)
            if new_anchors is not None:
                logger.warning("Anchor will be updated by Dataset's statistics")
                logger.warning(f"{anchor_generator.widths} -> {new_anchors[0]}")
                logger.warning(f"{anchor_generator.heights} -> {new_anchors[1]}")
                anchor_generator.widths = new_anchors[0]
                anchor_generator.heights = new_anchors[1]
                anchor_generator.gen_base_anchors()

    def _get_new_anchors(self, dataset: OTXDataset, anchor_generator: AnchorGenerator) -> tuple | None:
        """Get new anchors for SSD from OTXDataset."""
        from mmdet.datasets.transforms import Resize

        target_wh = None
        if isinstance(dataset.transforms, list):
            for transform in dataset.transforms:
                if isinstance(transform, Resize):
                    target_wh = transform.scale
        if target_wh is None:
            target_wh = (864, 864)
            msg = f"Cannot get target_wh from the dataset. Assign it with the default value: {target_wh}"
            logger.warning(msg)
        group_as = [len(width) for width in anchor_generator.widths]
        wh_stats = self._get_sizes_from_dataset_entity(dataset, list(target_wh))

        if len(wh_stats) < sum(group_as):
            logger.warning(
                f"There are not enough objects to cluster: {len(wh_stats)} were detected, while it should be "
                f"at least {sum(group_as)}. Anchor box clustering was skipped.",
            )
            return None

        return self._get_anchor_boxes(wh_stats, group_as)

    @staticmethod
    def _get_sizes_from_dataset_entity(dataset: OTXDataset, target_wh: list[int]) -> list[tuple[int, int]]:
        """Function to get width and height size of items in OTXDataset.

        Args:
            dataset(OTXDataset): OTXDataset in which to get statistics
            target_wh(list[int]): target width and height of the dataset
        Return
            list[tuple[int, int]]: tuples with width and height of each instance
        """
        wh_stats: list[tuple[int, int]] = []
        for item in dataset.dm_subset:
            for ann in item.annotations:
                if isinstance(ann, Bbox):
                    x1, y1, x2, y2 = ann.points
                    x1 = x1 / item.media.size[1] * target_wh[0]
                    y1 = y1 / item.media.size[0] * target_wh[1]
                    x2 = x2 / item.media.size[1] * target_wh[0]
                    y2 = y2 / item.media.size[0] * target_wh[1]
                    wh_stats.append((x2 - x1, y2 - y1))
        return wh_stats

    @staticmethod
    def _get_anchor_boxes(wh_stats: list[tuple[int, int]], group_as: list[int]) -> tuple:
        """Get new anchor box widths & heights using KMeans."""
        from sklearn.cluster import KMeans

        kmeans = KMeans(init="k-means++", n_clusters=sum(group_as), random_state=0).fit(wh_stats)
        centers = kmeans.cluster_centers_

        areas = np.sqrt(np.prod(centers, axis=1))
        idx = np.argsort(areas)

        widths = centers[idx, 0]
        heights = centers[idx, 1]

        group_as = np.cumsum(group_as[:-1])
        widths, heights = np.split(widths, group_as), np.split(heights, group_as)
        widths = [width.tolist() for width in widths]
        heights = [height.tolist() for height in heights]
        return widths, heights

    @staticmethod
    def get_classification_layers(
        config: DictConfig,
        model_registry: Registry,
        prefix: str,
    ) -> dict[str, dict[str, bool | int]]:
        """Return classification layer names by comparing two different number of classes models.

        Args:
            config (DictConfig): Config for building model.
            model_registry (Registry): Registry for building model.
            prefix (str): Prefix of model param name.
                Normally it is "model." since OTXModel set it's nn.Module model as self.model

        Return:
            dict[str, dict[str, int]]
            A dictionary contain classification layer's name and information.
            `use_bg` means whether SSD use background class. It if True if SSD use softmax loss, and
            it is False if SSD use cross entropy loss.
            `num_anchors` means number of anchors of layer. SSD have classification per each anchor,
            so we have to update every anchors.
        """
        sample_config = deepcopy(config)
        modify_num_classes(sample_config, 3)
        sample_model_dict = build_mm_model(sample_config, model_registry, None).state_dict()

        modify_num_classes(sample_config, 4)
        incremental_model_dict = build_mm_model(sample_config, model_registry, None).state_dict()

        classification_layers = {}
        for key in sample_model_dict:
            if sample_model_dict[key].shape != incremental_model_dict[key].shape:
                sample_model_dim = sample_model_dict[key].shape[0]
                if sample_model_dim % 3 != 0:
                    use_bg = True
                    num_anchors = int(sample_model_dim / 4)
                    classification_layers[prefix + key] = {"use_bg": use_bg, "num_anchors": num_anchors}
                else:
                    use_bg = False
                    num_anchors = int(sample_model_dim / 3)
                    classification_layers[prefix + key] = {"use_bg": use_bg, "num_anchors": num_anchors}
        return classification_layers

    def state_dict(self, *args, **kwargs) -> dict[str, Any]:
        """Return state dictionary of model entity with anchor information.

        Returns:
            A dictionary containing SSD state.

        """
        state_dict = super().state_dict(*args, **kwargs)
        anchor_generator = self.model.bbox_head.anchor_generator
        anchors = {"heights": anchor_generator.heights, "widths": anchor_generator.widths}
        state_dict["model.model.anchors"] = anchors
        return state_dict

    def load_state_dict_pre_hook(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs) -> None:
        """Modify input state_dict according to class name matching before weight loading."""
        model2ckpt = self.map_class_names(self.model_classes, self.ckpt_classes)

        for param_name, info in self.classification_layers.items():
            model_param = self.state_dict()[param_name].clone()
            ckpt_param = state_dict[prefix + param_name]
            use_bg = info["use_bg"]
            num_anchors = info["num_anchors"]
            if use_bg:
                num_ckpt_classes = len(self.ckpt_classes) + 1
                num_model_classes = len(self.model_classes) + 1
            else:
                num_ckpt_classes = len(self.ckpt_classes)
                num_model_classes = len(self.model_classes)

            for anchor_idx in range(num_anchors):
                for model_dst, ckpt_dst in enumerate(model2ckpt):
                    if ckpt_dst >= 0:
                        # Copying only matched weight rows
                        model_param[anchor_idx * num_model_classes + model_dst].copy_(
                            ckpt_param[anchor_idx * num_ckpt_classes + ckpt_dst],
                        )
                if use_bg:
                    model_param[anchor_idx * num_model_classes + num_model_classes - 1].copy_(
                        ckpt_param[anchor_idx * num_ckpt_classes + num_ckpt_classes - 1],
                    )

            # Replace checkpoint weight by mixed weights
            state_dict[prefix + param_name] = model_param

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Parameters for an exporter."""
        export_params = super()._export_parameters
        export_params["deploy_cfg"] = "otx.algo.detection.mmdeploy.ssd_mobilenetv2"
        export_params["input_size"] = self.image_size
        export_params["resize_mode"] = "standard"
        export_params["pad_value"] = 0
        export_params["swap_rgb"] = False

        return export_params

    def _set_anchors_hook(self, state_dict: dict[str, Any], *args, **kwargs) -> None:
        """Pre hook for pop anchor statistics from checkpoint state_dict."""
        anchors = state_dict.pop("model.model.anchors", None)
        if anchors is not None:
            anchor_generator = self.model.bbox_head.anchor_generator
            anchor_generator.widths = anchors["widths"]
            anchor_generator.heights = anchors["heights"]
            anchor_generator.gen_base_anchors()

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_ssd_ckpt(state_dict, add_prefix)
