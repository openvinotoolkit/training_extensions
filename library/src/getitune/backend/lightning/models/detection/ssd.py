# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) OpenMMLab. All rights reserved.

"""SSD object detector for the getitune detection.

Implementation modified from mmdet.models.detectors.single_stage.
Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/detectors/single_stage.py
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
from torch.export import Dim

from getitune.backend.lightning.exporter.base import ModelExporter
from getitune.backend.lightning.exporter.native import LightningModelExporter
from getitune.backend.lightning.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from getitune.backend.lightning.models.common.utils.assigners import MaxIoUAssigner
from getitune.backend.lightning.models.common.utils.coders import DeltaXYWHBBoxCoder
from getitune.backend.lightning.models.detection.base import LightningDetectionModel
from getitune.backend.lightning.models.detection.detectors import SingleStageDetector
from getitune.backend.lightning.models.detection.heads import SSDHead
from getitune.backend.lightning.models.detection.losses import SSDCriterion
from getitune.backend.lightning.models.detection.utils.prior_generators import SSDAnchorGeneratorClustered
from getitune.config.data import TileConfig
from getitune.metrics.fmeasure import MeanAveragePrecisionFMeasureCallable

if TYPE_CHECKING:
    import torch
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from torch import nn

    from getitune.backend.lightning.schedulers import LRSchedulerListCallable
    from getitune.data.dataset.base import VisionDataset
    from getitune.metrics import MetricCallable
    from getitune.types import PathLike
    from getitune.types.label import LabelInfoTypes


logger = logging.getLogger()


class SSD(LightningDetectionModel):
    """getitune Detection model class for SSD.

    Attributes:
        pretrained_urls (ClassVar[dict[str, str]]): Dictionary containing URLs for pretrained weights.

    Args:
        label_info (LabelInfoTypes): Information about the labels.
        data_input_params (DataInputParams | dict | None, optional): Parameters for image preprocessing.
            This parameter contains image input size, mean, and std, that is used to preprocess the input image.
            If None is given, default parameters for the specific model will be used.
            In most cases you don't need to set this parameter unless you change the image size or pretrained weights.
            Defaults to None.
        model_name (str, optional): Name of the model to use. Defaults to "ssd_mobilenetv2".
        optimizer (OptimizerCallable, optional): Callable for the optimizer. Defaults to DefaultOptimizerCallable.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): Callable for the learning rate scheduler.
            Defaults to DefaultSchedulerCallable.
        metric (MetricCallable, optional): Callable for the metric. Defaults to MeanAveragePrecisionFMeasureCallable.
        torch_compile (bool, optional): Whether to use torch compile. Defaults to False.
        tile_config (TileConfig, optional): Configuration for tiling. Defaults to TileConfig(enable_tiler=False).
        pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
        pretrained_weights (PathLike | None, optional): Path to the pretrained weights file. When None is passed,
            the default pretrained weights will be utilized for fine-tuning. Defaults to None.
    """

    pretrained_urls: ClassVar[dict[str, str]] = {
        "ssd_mobilenetv2": "https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/"
        "object_detection/v2/mobilenet_v2-2s_ssd-992x736.pth",
    }

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams | dict | None = None,
        model_name: Literal["ssd_mobilenetv2"] = "ssd_mobilenetv2",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MeanAveragePrecisionFMeasureCallable,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
        pretrained: bool = True,
        pretrained_weights: PathLike | None = None,
        export_nms: bool = False,
    ) -> None:
        super().__init__(
            label_info=label_info,
            data_input_params=data_input_params,
            model_name=model_name,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            tile_config=tile_config,
            pretrained=pretrained,
            pretrained_weights=pretrained_weights,
            export_nms=export_nms,
        )

    def _create_model(self, num_classes: int | None = None) -> SingleStageDetector:
        num_classes = num_classes if num_classes is not None else self.num_classes
        train_cfg = {
            "assigner": MaxIoUAssigner(
                min_pos_iou=0.0,
                ignore_iof_thr=-1,
                gt_max_assign_all=False,
                pos_iou_thr=0.4,
                neg_iou_thr=0.4,
            ),
            "allowed_border": -1,
            "pos_weight": -1,
            "debug": False,
            "use_giou": False,
            "use_focal": False,
        }
        test_cfg = {
            "nms": {"type": "nms", "iou_threshold": 0.45},
            "min_bbox_size": 0,
            "score_thr": 0.02,
            "max_per_img": 200,
        }
        backbone = self._build_backbone(model_name=self.model_name)
        bbox_head = SSDHead(
            model_name=self.model_name,
            num_classes=num_classes,
            anchor_generator=SSDAnchorGeneratorClustered(
                strides=[16, 32],
                widths=[
                    [38.641007923271076, 92.49516032784699, 271.4234764938237, 141.53469410876247],
                    [206.04136086566515, 386.6542727907841, 716.9892752215089, 453.75609561761405, 788.4629155558277],
                ],
                heights=[
                    [48.9243877087132, 147.73088476194903, 158.23569788707474, 324.14510379107367],
                    [587.6216059488938, 381.60024152086544, 323.5988913027747, 702.7486097568518, 741.4865860938451],
                ],
            ),
            bbox_coder=DeltaXYWHBBoxCoder(
                target_means=(0.0, 0.0, 0.0, 0.0),
                target_stds=(0.1, 0.1, 0.2, 0.2),
            ),
            init_cfg={
                "type": "Xavier",
                "layer": "Conv2d",
                "distribution": "uniform",
            },  # TODO (sungchul, kirill): remove
            train_cfg=train_cfg,  # TODO (sungchul, kirill): remove
            test_cfg=test_cfg,  # TODO (sungchul, kirill): remove
        )
        criterion = SSDCriterion(
            num_classes=num_classes,
            bbox_coder=DeltaXYWHBBoxCoder(
                target_means=(0.0, 0.0, 0.0, 0.0),
                target_stds=(0.1, 0.1, 0.2, 0.2),
            ),
        )
        model = SingleStageDetector(
            backbone=backbone,
            bbox_head=bbox_head,
            criterion=criterion,
            train_cfg=train_cfg,  # TODO (sungchul, kirill): remove
            test_cfg=test_cfg,  # TODO (sungchul, kirill): remove
        )
        model.init_weights()

        return model

    def _build_backbone(self, model_name: str) -> nn.Module:
        if "mobilenetv2" in model_name:
            from getitune.backend.lightning.models.common.backbones import build_model_including_pytorchcv

            return build_model_including_pytorchcv(
                cfg={
                    "type": "mobilenetv2_w1",
                    "out_indices": [4, 5],
                    "frozen_stages": -1,
                    "norm_eval": False,
                },
            )

        msg = f"Unknown backbone name: {model_name}"
        raise ValueError(msg)

    def setup(self, stage: str) -> None:
        """Callback for setup getitune SSD Model.

        SSD requires auto anchor generating w.r.t. training dataset for better accuracy.
        This callback will provide training dataset to model's anchor generator.

        Args:
            trainer(Trainer): Lightning trainer contains LightningModel and DataModule.
        """
        super().setup(stage=stage)

        if stage == "fit":
            anchor_generator = self.model.bbox_head.anchor_generator
            dataset = self.trainer.datamodule.train_dataloader().dataset
            new_anchors = self._get_new_anchors(dataset, anchor_generator)
            if new_anchors is not None:
                logger.warning("Anchor will be updated by Dataset's statistics")
                logger.warning(f"{anchor_generator.widths} -> {new_anchors[0]}")
                logger.warning(f"{anchor_generator.heights} -> {new_anchors[1]}")
                anchor_generator.widths = new_anchors[0]
                anchor_generator.heights = new_anchors[1]
                anchor_generator.gen_base_anchors()
                self.hparams["ssd_anchors"] = {
                    "heights": anchor_generator.heights,
                    "widths": anchor_generator.widths,
                }

    def _get_new_anchors(self, dataset: VisionDataset, anchor_generator: SSDAnchorGeneratorClustered) -> tuple | None:
        """Get new anchors for SSD from VisionDataset."""
        from torchvision.transforms.v2._container import Compose

        from getitune.data.augmentation.transforms import Resize

        target_wh = None
        if isinstance(dataset.transforms, Compose):
            for transform in dataset.transforms.transforms:
                if isinstance(transform, Resize):
                    target_wh = transform.scale
        if target_wh is None:
            if self.data_input_params.input_size is None:
                msg = "input_size should not be None."
                raise ValueError(msg)
            target_wh = list(reversed(self.data_input_params.input_size))  # type: ignore[assignment]
            msg = f"Cannot get target_wh from the dataset. Assign it with the default value: {target_wh}"
            logger.warning(msg)
        group_as = [len(width) for width in anchor_generator.widths]
        wh_stats = self._get_sizes_from_dataset_entity(dataset, list(target_wh))  # type: ignore[arg-type]

        if len(wh_stats) < sum(group_as):
            logger.warning(
                f"There are not enough objects to cluster: {len(wh_stats)} were detected, while it should be "
                f"at least {sum(group_as)}. Anchor box clustering was skipped.",
            )
            return None

        return self._get_anchor_boxes(wh_stats, group_as)

    @staticmethod
    def _get_sizes_from_dataset_entity(dataset: VisionDataset, target_wh: list[int]) -> np.ndarray:
        """Function to get width and height size of items in VisionDataset.

        Args:
            dataset(VisionDataset): VisionDataset in which to get statistics
            target_wh(list[int]): target width and height of the dataset
        Return
            list[tuple[int, int]]: tuples with width and height of each instance
        """
        wh_stats = np.empty((0, 2), dtype=np.float32)
        for item in dataset.dm_subset:
            height, width = item.img_info.img_shape
            x1 = item.bboxes[:, 0]
            y1 = item.bboxes[:, 1]
            x2 = item.bboxes[:, 2]
            y2 = item.bboxes[:, 3]

            w = (x2 - x1) / width * target_wh[0]
            h = (y2 - y1) / height * target_wh[1]

            wh_stats = np.concatenate((wh_stats, np.stack((w, h), axis=1)), axis=0)
        return wh_stats

    @staticmethod
    def _get_anchor_boxes(wh_stats: np.ndarray, group_as: list[int]) -> tuple:
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

    def _identify_classification_layers(  # type: ignore[override]
        self,
        prefix: str = "model.",
    ) -> dict[str, dict[str, bool | int]]:
        """Return classification layer names by comparing two different number of classes models.

        Args:
            prefix (str): Prefix of model param name.
                Normally it is "model." since LightningModel set it's nn.Module model as self.model

        Return:
            dict[str, dict[str, int]]
            A dictionary contain classification layer's name and information.
            `use_bg` means whether SSD use background class. It if True if SSD use softmax loss, and
            it is False if SSD use cross entropy loss.
            `num_anchors` means number of anchors of layer. SSD have classification per each anchor,
            so we have to update every anchors.
        """
        sample_model_dict = self._create_model(num_classes=3).state_dict()
        incremental_model_dict = self._create_model(num_classes=4).state_dict()

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

    def load_state_dict_pre_hook(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs) -> None:
        """Modify input state_dict according to class name matching. It is used for incremental learning."""
        model2ckpt = self.map_class_names(self.model_classes, self.ckpt_classes)

        for param_name, info in self._identify_classification_layers().items():
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
    def _exporter(self) -> ModelExporter:
        """Creates ModelExporter object that can export the model."""
        return LightningModelExporter(
            task_level_export_parameters=self._export_parameters,
            data_input_params=self.data_input_params,
            resize_mode="standard",
            pad_value=0,
            swap_rgb=False,
            via_onnx=False,
            onnx_export_configuration={
                "input_names": ["image"],
                "output_names": ["boxes", "labels"],
                "dynamic_shapes": {"inputs": {0: Dim("batch")}},
                "autograd_inlining": False,
            },
            output_names=["bboxes", "labels", "feature_vector", "saliency_map"] if self.explain_mode else None,
        )

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Callback on load checkpoint."""
        if (hparams := checkpoint.get("hyper_parameters")) and (anchors := hparams.get("ssd_anchors", None)):
            anchor_generator = self.model.bbox_head.anchor_generator
            anchor_generator.widths = anchors["widths"]
            anchor_generator.heights = anchors["heights"]
            anchor_generator.gen_base_anchors()

        return super().on_load_checkpoint(checkpoint)

    @property
    def _default_preprocessing_params(self) -> DataInputParams | dict[str, DataInputParams]:
        return DataInputParams(input_size=(864, 864), mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
