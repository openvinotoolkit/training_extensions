# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Main class for OTX MaskDINO model."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch
from torch import Tensor, nn
from torchvision import tv_tensors
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import box_convert

from otx.algo.instance_segmentation.heads import MaskDINODecoderHead, MaskDINOEncoderHead, MaskDINOHead
from otx.algo.instance_segmentation.losses import MaskDINOCriterion
from otx.algo.instance_segmentation.segmentors import MaskDINOModule
from otx.algo.instance_segmentation.utils.utils import ShapeSpec
from otx.algo.modules.norm import AVAILABLE_NORMALIZATION_LIST, FrozenBatchNorm2d
from otx.algo.utils.mmengine_utils import load_from_http
from otx.core.config.data import TileConfig
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.instance_segmentation import InstanceSegBatchDataEntity, InstanceSegBatchPredEntity
from otx.core.data.entity.utils import stack_batch
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.metrics.mean_ap import MaskRLEMeanAPFMeasureCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.instance_segmentation import ExplainableOTXInstanceSegModel
from otx.core.utils.mask_util import polygon_to_bitmap

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallable
    from otx.core.schedulers import LRSchedulerListCallable
    from otx.core.types.label import LabelInfoTypes


class MaskDINO(ExplainableOTXInstanceSegModel):
    """OTX MaskDINO Instance Segmentation model."""

    backbone_cfg: ClassVar[dict[str, Any]] = {
        "resnet50": {
            "backbone": resnet50(norm_layer=FrozenBatchNorm2d),
            "return_layers": {
                "layer1": "res2",
                "layer2": "res3",
                "layer3": "res4",
                "layer4": "res5",
            },
            "strides": [4, 8, 16, 32],
            "channels": [256, 512, 1024, 2048],
            "weights": (
                "https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/"
                "maskdino_r50_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask46.3ap_box51.7ap.pth"
            ),
        },
    }

    mean: tuple[float, float, float] = (123.675, 116.28, 103.53)
    std: tuple[float, float, float] = (58.395, 57.12, 57.375)

    def __init__(
        self,
        model_name: Literal["resnet50"],
        label_info: LabelInfoTypes,
        input_size: tuple[int, int] = (1024, 1024),
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MaskRLEMeanAPFMeasureCallable,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
    ):
        self.load_from: str = self.backbone_cfg[model_name]["weights"]
        super().__init__(
            model_name=model_name,
            label_info=label_info,
            input_size=input_size,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            tile_config=tile_config,
        )

    def _build_fmap_shape_specs(
        self,
        out_features: list[str],
        strides: list[int],
        channels: list[int],
    ) -> dict[str, ShapeSpec]:
        """Build feature map shape specs frm backbone config.

        Args:
            out_features (list[str]): feature map names.
            strides (list[int]): stride of each feature map.
            channels (list[int]): number of out channels for each feature map.

        Returns:
            dict[str, ShapeSpec]: feature map shape specs.
        """
        output_shape_dict = {}
        for out_feature, stride, channel in zip(out_features, strides, channels, strict=True):
            output_shape_dict[out_feature] = ShapeSpec(
                channels=channel,
                stride=stride,
            )
        return output_shape_dict

    def _build_backbone(self) -> tuple[nn.Module, dict[str, ShapeSpec]]:
        """Build backbone.

        Raises:
            ValueError: If the backbone is not supported.

        Returns:
            tuple[nn.Module, dict[str, ShapeSpec]]: Backbone and shape specs.
        """
        if self.model_name in self.backbone_cfg:
            backbone_cfg = self.backbone_cfg[self.model_name]

            backbone = IntermediateLayerGetter(
                backbone_cfg["backbone"],
                return_layers=backbone_cfg["return_layers"],
            )

            shape_spec = self._build_fmap_shape_specs(
                out_features=list(backbone_cfg["return_layers"].values()),
                strides=backbone_cfg["strides"],
                channels=backbone_cfg["channels"],
            )

            return backbone, shape_spec
        msg = f"Backbone {self.model_name} is not supported."
        raise ValueError(msg)

    def _build_model(self, num_classes: int) -> MaskDINOModule:
        """Build MaskDINO model.

        Args:
            num_classes (int): Number of classes.

        Returns:
            MaskDINOModule: MaskDINO model.
        """
        backbone, fmap_shape_specs = self._build_backbone()
        model_name = self.model_name

        pixel_decoder = MaskDINOEncoderHead(model_name=model_name, input_shape=fmap_shape_specs)
        predictor = MaskDINODecoderHead(model_name=model_name, num_classes=num_classes)

        sem_seg_head = MaskDINOHead(
            num_classes=num_classes,
            pixel_decoder=pixel_decoder,
            predictor=predictor,
            num_queries=300,
            test_topk_per_image=100,
        )

        # building criterion
        criterion = MaskDINOCriterion(
            num_classes=num_classes,
        )

        return MaskDINOModule(
            backbone=backbone,
            sem_seg_head=sem_seg_head,
            criterion=criterion,
        )

    def _create_model(self) -> nn.Module:
        """Create MaskDINO model and load pre-trained weights.

        Detectron2 state dict have different layer structure than torchvision state dict,
        so we need to implement custom loading logic.

        Returns:
            nn.Module: MaskDINO model.
        """
        # TODO(Eugene): make it more general, now it only supports R50.

        # Firstly load all weights on MaskDINO heads.
        detector = super()._create_model()

        # Then, load pre-trained backbone weights.
        pretrained = load_from_http(self.load_from, map_location="cpu")
        pretrained = pretrained.pop("model")
        backbone_weights = []
        backbone_shortcut_weights = []
        for layer_name, weights in pretrained.items():
            if layer_name.startswith("backbone"):
                if "shortcut" in layer_name:
                    backbone_shortcut_weights.append(weights)
                else:
                    backbone_weights.append(weights)

        # Load pre-trained backbone weights to TV backbone weights.
        tv_model_dict = {}
        for layer_name in detector.backbone.state_dict():
            w = backbone_shortcut_weights.pop(0) if "downsample" in layer_name else backbone_weights.pop(0)
            tv_model_dict[layer_name] = w.clone()
        detector.backbone.load_state_dict(tv_model_dict)

        # load conv modules weight
        conv_modules = [
            "sem_seg_head.pixel_decoder.mask_features",
            "sem_seg_head.pixel_decoder.adapter_1",
            "sem_seg_head.pixel_decoder.layer_1",
        ]
        conv_module_dict = {}
        for conv_module in conv_modules:
            weight_name = f"{conv_module}.weight"
            bias_name = f"{conv_module}.bias"
            norm_name = f"{conv_module}.norm.weight"
            norm_bias_name = f"{conv_module}.norm.bias"

            if weight_name in pretrained:
                new_weight_name = f"{conv_module}.conv.weight"
                conv_module_dict[new_weight_name] = pretrained[weight_name].clone()
            if bias_name in pretrained:
                new_bias_name = f"{conv_module}.conv.bias"
                conv_module_dict[new_bias_name] = pretrained[bias_name].clone()
            if norm_name in pretrained:
                new_norm_name = f"{conv_module}.gn.weight"
                conv_module_dict[new_norm_name] = pretrained[norm_name].clone()
            if norm_bias_name in pretrained:
                new_norm_bias_name = f"{conv_module}.gn.bias"
                conv_module_dict[new_norm_bias_name] = pretrained[norm_bias_name].clone()

        incompatible_keys = detector.load_state_dict(conv_module_dict, strict=False)
        if len(incompatible_keys.unexpected_keys):
            msg = f"Unexpected keys: {incompatible_keys.unexpected_keys}"
            raise ValueError(msg)
        return detector

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        if self.input_size is None:
            msg = f"Input size attribute is not set for {self.__class__}"
            raise ValueError(msg)

        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=(1, 3, *self.input_size),
            mean=self.mean,
            std=self.std,
            resize_mode="standard",
            swap_rgb=False,
            via_onnx=True,
            onnx_export_configuration={
                "input_names": ["image"],
                "dynamic_axes": {
                    "image": {0: "batch", 2: "height", 3: "width"},
                    "boxes": {0: "batch", 1: "num_dets"},
                    "labels": {0: "batch", 1: "num_dets"},
                    "masks": {0: "batch", 1: "num_dets", 2: "height", 3: "width"},
                },
                "opset_version": 16,
                "autograd_inlining": False,
            },
            output_names=["bboxes", "labels", "masks"],
        )

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for MaskDINO-R50."""
        return {"model_type": "transformer"}

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict[str, Any]]]:
        """Configure an optimizer and learning-rate schedulers."""
        param_groups = self._get_optim_params(self.model)
        optimizer = self.optimizer_callable(param_groups)

        schedulers = self.scheduler_callable(optimizer)

        def ensure_list(item: Any) -> list:  # noqa: ANN401
            return item if isinstance(item, list) else [item]

        lr_scheduler_configs = []
        for scheduler in ensure_list(schedulers):
            lr_scheduler_config = {"scheduler": scheduler}
            if hasattr(scheduler, "interval"):
                lr_scheduler_config["interval"] = scheduler.interval
            if hasattr(scheduler, "monitor"):
                lr_scheduler_config["monitor"] = scheduler.monitor
            lr_scheduler_configs.append(lr_scheduler_config)

        return [optimizer], lr_scheduler_configs

    def _get_optim_params(self, model: nn.Module) -> list[dict[str, Any]]:
        """Get optimizer parameters."""
        _optimizer = self.optimizer_callable(self.parameters())

        # Configurable from MaskDINO recipe
        base_lr = _optimizer.defaults["lr"]
        weight_decay = _optimizer.defaults["weight_decay"]
        defaults = {
            "lr": base_lr,
            "weight_decay": weight_decay,
        }

        # Static optimizer params
        weight_decay_norm = 0.0
        weight_decay_embed = 0.0
        backbone_multiplier = 0.1

        params = []
        uniques = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in uniques:
                    continue
                uniques.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * backbone_multiplier
                if "relative_position_bias_table" in module_param_name or "absolute_pos_embed" in module_param_name:
                    hyperparams["weight_decay"] = 0.0
                if module.__class__ in AVAILABLE_NORMALIZATION_LIST:
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})
        return params

    def _customize_inputs(self, entity: InstanceSegBatchDataEntity) -> dict[str, Any]:
        """Customize inputs for MaskDINO model.

        Args:
            entity (InstanceSegBatchDataEntity): InstanceSegBatchDataEntity object.

        Returns:
            dict[str, Any]: Customized inputs for MaskDINO model.
        """
        if isinstance(entity.images, list):
            entity.images, entity.imgs_info = stack_batch(entity.images, entity.imgs_info, pad_size_divisor=32)
        img_shapes = [img_info.img_shape for img_info in entity.imgs_info]
        images = ImageList(entity.images, img_shapes)

        if self.training:
            targets = []
            for img_info, bboxes, labels, polygons, gt_masks in zip(
                entity.imgs_info,
                entity.bboxes,
                entity.labels,
                entity.polygons,
                entity.masks,
            ):
                masks = polygon_to_bitmap(polygons, *img_info.img_shape) if len(polygons) else gt_masks
                norm_shape = torch.tile(torch.tensor(img_info.img_shape, device=img_info.device), (2,))
                targets.append(
                    {
                        "boxes": box_convert(bboxes, in_fmt="xyxy", out_fmt="cxcywh") / norm_shape,
                        "labels": labels,
                        "masks": tv_tensors.Mask(masks, device=img_info.device, dtype=torch.bool),
                    },
                )

        return {
            "images": images,
            "imgs_info": entity.imgs_info,
            "targets": targets if self.training else None,
        }

    def _customize_outputs(
        self,
        outputs: dict[str, Tensor]  # type: ignore[override]
        | tuple[list[Tensor], list[torch.LongTensor], list[tv_tensors.Mask]],
        inputs: InstanceSegBatchDataEntity,
    ) -> OTXBatchLossEntity | InstanceSegBatchPredEntity:
        """Customize outputs for MaskDINO model.

        Args:
            outputs (dict[str, Tensor] | tuple[list[Tensor], list[torch.LongTensor], list[tv_tensors.Mask]):
                dict[str, Tensor]: dictionary of losses in training mode.
                tuple[list[Tensor], list[torch.LongTensor], list[tv_tensors.Mask]: tuple of outputs in inference mode.
                    list[Tensor]: bounding boxes and scores with shape [N, 5]
                    list[torch.LongTensor]: labels with shape [N]
                    list[tv_tensors.Mask]: masks with shape [N, H, W]
            inputs (InstanceSegBatchDataEntity): InstanceSegBatchDataEntity object.

        Raises:
            NotImplementedError: If explain mode is not supported yet.

        Returns:
            OTXBatchLossEntity | InstanceSegBatchPredEntity: Customized outputs for MaskDINO model.
        """
        if self.training and isinstance(outputs, dict):
            return sum(outputs.values())  # type: ignore[return-value]

        batch_bboxes_scores: list[Tensor]
        batch_labels: list[torch.LongTensor]
        batch_masks: list[tv_tensors.Mask]
        batch_bboxes_scores, batch_labels, batch_masks = outputs  # type: ignore[assignment]

        batch_bboxes = [bboxes_scores[:, :4] for bboxes_scores in batch_bboxes_scores]
        batch_scores = [bboxes_scores[:, 4] for bboxes_scores in batch_bboxes_scores]

        if self.explain_mode:
            msg = "Explain mode is not supported yet."
            raise NotImplementedError(msg)

        return InstanceSegBatchPredEntity(
            batch_size=inputs.batch_size,
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=batch_scores,
            bboxes=batch_bboxes,
            masks=batch_masks,
            polygons=[],
            labels=batch_labels,
        )
