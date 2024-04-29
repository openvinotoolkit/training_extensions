# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""SSD object detector for the OTX detection."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from datumaro.components.annotation import Bbox
from mmengine.structures import InstanceData
from torch import nn
from torchvision import tv_tensors

from otx.algo.detection.backbones.pytorchcv_backbones import _build_model_including_pytorchcv
from otx.algo.detection.heads.ssd_head import SSDHead
from otx.algo.modules.base_module import BaseModule
from otx.algo.utils.mmconfig import read_mmconfig
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.config.data import TileConfig
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.detection import DetBatchDataEntity, DetBatchPredEntity
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.metrics.mean_ap import MeanAPCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.detection import ExplainableOTXDetModel
from otx.core.model.utils.mmdet import DetDataPreprocessor
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.label import LabelInfoTypes
from otx.core.utils.build import modify_num_classes
from otx.core.utils.config import convert_conf_to_mmconfig_dict, inplace_num_classes
from otx.core.utils.utils import get_mean_std_from_data_processing

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from mmengine import ConfigDict
    from omegaconf import DictConfig
    from torch import Tensor

    from otx.algo.detection.heads.custom_anchor_generator import SSDAnchorGeneratorClustered
    from otx.core.data.dataset.base import OTXDataset
    from otx.core.metrics import MetricCallable


logger = logging.getLogger()


# This class and its supporting functions below lightly adapted from the mmdet SingleStageDetector available at:
# https://github.com/open-mmlab/mmdetection/blob/cfd5d3a985b0249de009b67d04f37263e11cdf3d/mmdet/models/detectors/single_stage.py
class SingleStageDetector(BaseModule):
    """Single stage detector implementation from mmdet."""

    def __init__(
        self,
        backbone: ConfigDict | dict,
        bbox_head: ConfigDict | dict,
        data_preprocessor: ConfigDict | dict,
        train_cfg: ConfigDict | dict | None = None,
        test_cfg: ConfigDict | dict | None = None,
        init_cfg: ConfigDict | list[ConfigDict] | dict | list[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.backbone = self.build_backbone(backbone)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = self.build_bbox_head(bbox_head)
        self.data_preprocessor = self.build_det_data_preprocessor(data_preprocessor)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def build_backbone(self, cfg: ConfigDict | dict) -> nn.Module:
        """Build backbone."""
        return _build_model_including_pytorchcv(cfg)

    def build_bbox_head(self, cfg: ConfigDict | dict) -> nn.Module:
        """Build bbox head."""
        return SSDHead(**cfg)

    def build_det_data_preprocessor(self, cfg: ConfigDict | dict | nn.Module) -> nn.Module:
        """Build DetDataPreprocessor.

        TODO (someone): DetDataPreprocessor will be removed.
        """
        return DetDataPreprocessor(**cfg)

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str] | str,
        unexpected_keys: list[str] | str,
        error_msgs: list[str] | str,
    ) -> None:
        """Exchange bbox_head key to rpn_head key.

        When loading two-stage weights into single-stage model.
        """
        bbox_head_prefix = prefix + ".bbox_head" if prefix else "bbox_head"
        bbox_head_keys = [k for k in state_dict if k.startswith(bbox_head_prefix)]
        rpn_head_prefix = prefix + ".rpn_head" if prefix else "rpn_head"
        rpn_head_keys = [k for k in state_dict if k.startswith(rpn_head_prefix)]
        if len(bbox_head_keys) == 0 and len(rpn_head_keys) != 0:
            for rpn_head_key in rpn_head_keys:
                bbox_head_key = bbox_head_prefix + rpn_head_key[len(rpn_head_prefix) :]
                state_dict[bbox_head_key] = state_dict.pop(rpn_head_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        entity: DetBatchDataEntity,
        mode: str = "tensor",
    ) -> dict[str, torch.Tensor] | list[InstanceData] | tuple[torch.Tensor] | torch.Tensor:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`InstanceData`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle either back propagation or
        parameter update, which are supposed to be done in :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`InstanceData`], optional): A batch of
                data samples that contain annotations and predictions.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`InstanceData`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == "loss":
            return self.loss(entity)
        if mode == "predict":
            return self.predict(entity)
        if mode == "tensor":
            return self._forward(entity)

        msg = f"Invalid mode {mode}. Only supports loss, predict and tensor mode"
        raise RuntimeError(msg)

    def loss(
        self,
        entity: DetBatchDataEntity,
    ) -> dict | list:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`InstanceData`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(entity.images)
        return self.bbox_head.loss(x, entity)

    def predict(
        self,
        entity: DetBatchDataEntity,
        rescale: bool = True,
    ) -> list[InstanceData]:
        """Predict results from a batch of inputs and data samples with post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`InstanceData`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Detection results of the
            input images. Each InstanceData usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        x = self.extract_feat(entity.images)
        return self.bbox_head.predict(x, entity, rescale=rescale)

    def export(
        self,
        batch_inputs: Tensor,
        batch_img_metas: list[dict],
        rescale: bool = True,
        explain_mode: bool = False,
    ) -> list[InstanceData] | dict:
        """Predict results from a batch of inputs and data samples with post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`InstanceData`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Detection results of the
            input images. Each InstanceData usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        if explain_mode:
            backbone_feat = self.extract_feat(batch_inputs)
            bbox_head_feat = self.bbox_head.forward(backbone_feat)
            feature_vector = self.feature_vector_fn(backbone_feat)
            saliency_map = self.explain_fn(bbox_head_feat[0])
            bboxes, labels = self.bbox_head.export(backbone_feat, batch_img_metas, rescale=rescale)
            return {
                "bboxes": bboxes,
                "labels": labels,
                "feature_vector": feature_vector,
                "saliency_map": saliency_map,
            }
        x = self.extract_feat(batch_inputs)
        return self.bbox_head.export(x, batch_img_metas, rescale=rescale)

    def _forward(
        self,
        batch_inputs: Tensor,
        batch_data_samples: list[InstanceData] | None = None,
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Network forward process.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`InstanceData`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x = self.extract_feat(batch_inputs)
        return self.bbox_head.forward(x)

    def extract_feat(self, batch_inputs: Tensor) -> tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    @property
    def with_neck(self) -> bool:
        """bool: whether the detector has a neck."""
        return hasattr(self, "neck") and self.neck is not None

    @property
    def with_shared_head(self) -> bool:
        """bool: whether the detector has a shared head in the RoI Head."""
        return hasattr(self, "roi_head") and self.roi_head.with_shared_head

    @property
    def with_bbox(self) -> bool:
        """bool: whether the detector has a bbox head."""
        return (hasattr(self, "roi_head") and self.roi_head.with_bbox) or (
            hasattr(self, "bbox_head") and self.bbox_head is not None
        )

    @property
    def with_mask(self) -> bool:
        """bool: whether the detector has a mask head."""
        return (hasattr(self, "roi_head") and self.roi_head.with_mask) or (
            hasattr(self, "mask_head") and self.mask_head is not None
        )


class SSD(ExplainableOTXDetModel):
    """Detecion model class for SSD."""

    def __init__(
        self,
        label_info: LabelInfoTypes,
        variant: Literal["mobilenetv2"],
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MeanAPCallable,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
    ) -> None:
        model_name = f"ssd_{variant}"
        config = read_mmconfig(model_name=model_name)
        config = inplace_num_classes(cfg=config, num_classes=self._dispatch_label_info(label_info).num_classes)
        self.config = config
        self.load_from = config.pop("load_from", None)
        super().__init__(
            label_info=label_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            tile_config=tile_config,
        )
        self.image_size = (1, 3, 864, 864)
        self.tile_image_size = self.image_size

    def _create_model(self) -> nn.Module:
        from mmengine.runner import load_checkpoint

        config = deepcopy(self.config)
        self.classification_layers = self.get_classification_layers(config, "model.")
        detector = SingleStageDetector(**convert_conf_to_mmconfig_dict(config))
        if self.load_from is not None:
            load_checkpoint(detector, self.load_from, map_location="cpu")
        return detector

    def _customize_inputs(self, entity: DetBatchDataEntity) -> dict[str, Any]:
        inputs: dict[str, Any] = {}

        inputs["entity"] = entity
        inputs["mode"] = "loss" if self.training else "predict"

        return inputs

    def _customize_outputs(
        self,
        outputs: list[InstanceData] | dict,
        inputs: DetBatchDataEntity,
    ) -> DetBatchPredEntity | OTXBatchLossEntity:
        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            for k, v in outputs.items():
                if isinstance(v, list):
                    losses[k] = sum(v)
                elif isinstance(v, torch.Tensor):
                    losses[k] = v
                else:
                    msg = "Loss output should be list or torch.tensor but got {type(v)}"
                    raise TypeError(msg)
            return losses

        scores = []
        bboxes = []
        labels = []
        predictions = outputs["predictions"] if isinstance(outputs, dict) else outputs
        for img_info, prediction in zip(inputs.imgs_info, predictions):
            if not isinstance(prediction, InstanceData):
                raise TypeError(prediction)
            scores.append(prediction.scores)
            bboxes.append(
                tv_tensors.BoundingBoxes(
                    prediction.bboxes,
                    format="XYXY",
                    canvas_size=img_info.ori_shape,
                ),
            )
            labels.append(prediction.labels)

        if self.explain_mode:
            if not isinstance(outputs, dict):
                msg = f"Model output should be a dict, but got {type(outputs)}."
                raise ValueError(msg)

            if "feature_vector" not in outputs:
                msg = "No feature vector in the model output."
                raise ValueError(msg)

            if "saliency_map" not in outputs:
                msg = "No saliency maps in the model output."
                raise ValueError(msg)

            saliency_map = outputs["saliency_map"].detach().cpu().numpy()
            feature_vector = outputs["feature_vector"].detach().cpu().numpy()

            return DetBatchPredEntity(
                batch_size=len(outputs),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=scores,
                bboxes=bboxes,
                labels=labels,
                saliency_map=saliency_map,
                feature_vector=feature_vector,
            )

        return DetBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            labels=labels,
        )

    def setup(self, stage: str) -> None:
        """Callback for setup OTX SSD Model.

        OTXSSD requires auto anchor generating w.r.t. training dataset for better accuracy.
        This callback will provide training dataset to model's anchor generator.

        Args:
            trainer(Trainer): Lightning trainer contains OTXLitModule and OTXDatamodule.
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

    def _get_new_anchors(self, dataset: OTXDataset, anchor_generator: SSDAnchorGeneratorClustered) -> tuple | None:
        """Get new anchors for SSD from OTXDataset."""
        from torchvision.transforms.v2._container import Compose

        from otx.core.data.transform_libs.torchvision import Resize

        target_wh = None
        if isinstance(dataset.transforms, Compose):
            for transform in dataset.transforms.transforms:
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
        sample_model_dict = SingleStageDetector(**convert_conf_to_mmconfig_dict(sample_config)).state_dict()

        modify_num_classes(sample_config, 4)
        incremental_model_dict = SingleStageDetector(**convert_conf_to_mmconfig_dict(sample_config)).state_dict()

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
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        if self.image_size is None:
            raise ValueError(self.image_size)

        mean, std = get_mean_std_from_data_processing(self.config)

        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=self.image_size,
            mean=mean,
            std=std,
            resize_mode="standard",
            pad_value=0,
            swap_rgb=False,
            via_onnx=True,  # Currently SSD should be exported through ONNX
            onnx_export_configuration={
                "input_names": ["image"],
                "output_names": ["boxes", "labels"],
                "dynamic_axes": {
                    "image": {0: "batch", 2: "height", 3: "width"},
                    "boxes": {0: "batch", 1: "num_dets"},
                    "labels": {0: "batch", 1: "num_dets"},
                },
                "autograd_inlining": False,
            },
            output_names=["bboxes", "labels", "feature_vector", "saliency_map"] if self.explain_mode else None,
        )

    def forward_for_tracing(self, inputs: Tensor) -> list[InstanceData]:
        """Forward function for export."""
        shape = (int(inputs.shape[2]), int(inputs.shape[3]))
        meta_info = {
            "pad_shape": shape,
            "batch_input_shape": shape,
            "img_shape": shape,
            "scale_factor": (1.0, 1.0),
        }
        meta_info_list = [meta_info] * len(inputs)
        return self.model.export(inputs, meta_info_list, explain_mode=self.explain_mode)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Callback on load checkpoint."""
        if (hparams := checkpoint.get("hyper_parameters")) and (anchors := hparams.get("ssd_anchors", None)):
            anchor_generator = self.model.bbox_head.anchor_generator
            anchor_generator.widths = anchors["widths"]
            anchor_generator.heights = anchors["heights"]
            anchor_generator.gen_base_anchors()

        return super().on_load_checkpoint(checkpoint)

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_ssd_ckpt(state_dict, add_prefix)
