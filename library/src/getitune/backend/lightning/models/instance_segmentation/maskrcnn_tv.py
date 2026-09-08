# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""TV MaskRCNN model implementations."""

# type: ignore[override]

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch
from torch import Tensor
from torchvision import tv_tensors
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, _default_anchorgen
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_V2_Weights, MaskRCNNPredictor

from getitune.backend.lightning.exporter.base import ModelExporter
from getitune.backend.lightning.exporter.native import LightningModelExporter
from getitune.backend.lightning.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from getitune.backend.lightning.models.instance_segmentation.base import LightningInstanceSegModel
from getitune.backend.lightning.models.instance_segmentation.heads import TVRoIHeads
from getitune.backend.lightning.models.instance_segmentation.segmentors.maskrcnn_tv import (
    FastRCNNConvFCHead,
    MaskRCNN,
    MaskRCNNBackbone,
    MaskRCNNHeads,
    RPNHead,
)
from getitune.config.data import TileConfig
from getitune.data.entity.base import BatchLoss
from getitune.data.entity.sample import PredictionBatch, SampleBatch
from getitune.data.entity.utils import stack_batch
from getitune.metrics.fmeasure import MaskRLEMeanAPFMeasureCallable

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from torch import nn

    from getitune.backend.lightning.schedulers import LRSchedulerListCallable
    from getitune.metrics import MetricCallable
    from getitune.types import PathLike
    from getitune.types.label import LabelInfoTypes


class MaskRCNNTV(LightningInstanceSegModel):
    """Implementation of torchvision MaskRCNN for instance segmentation.

    Args:
        label_info (LabelInfoTypes): Information about the labels used in the model.
        data_input_params (DataInputParams | dict | None, optional): Parameters for the image data preprocessing.
            If None is given, default parameters for the specific model will be used.
        model_name (str, optional): Name of the model. Defaults to "maskrcnn_resnet_50".
        optimizer (OptimizerCallable, optional): Optimizer for the model. Defaults to DefaultOptimizerCallable.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): Scheduler for the model.
            Defaults to DefaultSchedulerCallable.
        metric (MetricCallable, optional): Metric for evaluating the model.
            Defaults to MaskRLEMeanAPFMeasureCallable.
        torch_compile (bool, optional): Whether to use torch compile. Defaults to False.
        tile_config (TileConfig, optional): Configuration for tiling. Defaults to TileConfig(enable_tiler=False).
        pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
        pretrained_weights (PathLike | None, optional): Path to the pretrained weights file. When None is passed,
            the default pretrained weights will be utilized for fine-tuning. Defaults to None.
    """

    _nms_always_embedded: ClassVar[bool] = True
    pretrained_urls: ClassVar[dict[str, str]] = {
        "maskrcnn_resnet_50": MaskRCNN_ResNet50_FPN_V2_Weights.verify("DEFAULT").url,
    }

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams | dict | None = None,
        model_name: Literal["maskrcnn_resnet_50"] = "maskrcnn_resnet_50",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MaskRLEMeanAPFMeasureCallable,
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

    def _create_model(self, num_classes: int | None = None) -> MaskRCNN:
        num_classes = num_classes if num_classes is not None else self.num_classes

        # NOTE: Add 1 to num_classes to account for background class.
        num_classes = num_classes + 1

        # init model components, model itself and load weights
        rpn_anchor_generator = _default_anchorgen()
        backbone = MaskRCNNBackbone(model_name=self.model_name)
        rpn_head = RPNHead(model_name=self.model_name, anchorgen=rpn_anchor_generator)
        box_head = FastRCNNConvFCHead(model_name=self.model_name)
        mask_head = MaskRCNNHeads(model_name=self.model_name)

        model = MaskRCNN(
            backbone,
            num_classes=91,
            rpn_anchor_generator=rpn_anchor_generator,
            rpn_head=rpn_head,
            box_head=box_head,
            mask_head=mask_head,
        )

        # Replace RoIHeads since torchvision does not allow customized roi_heads.
        model.roi_heads = TVRoIHeads(
            model.roi_heads.box_roi_pool,
            model.roi_heads.box_head,
            model.roi_heads.box_predictor,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=model.roi_heads.score_thresh,
            nms_thresh=model.roi_heads.nms_thresh,
            detections_per_img=model.roi_heads.detections_per_img,
            mask_roi_pool=model.roi_heads.mask_roi_pool,
            mask_head=model.roi_heads.mask_head,
            mask_predictor=model.roi_heads.mask_predictor,
        )

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = model.roi_heads.mask_predictor.conv5_mask.out_channels

        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes,
        )

        return model

    def _move_model_tensors_to_device_and_dtype(self, device: str, dtype: torch.dtype | None = None) -> nn.Module:
        """Move model and torchvision-specific cached tensors to the specified device and dtype.

        Torchvision's AnchorGenerator caches cell_anchors which are not moved by model.to().
        This override ensures those cached tensors are also moved to avoid device/dtype mismatch
        during TorchScript tracing for export.

        Args:
            device: Target device ('cpu', 'cuda', etc.)
            dtype: Target dtype for floating-point tensors. If None, dtype is not changed.
                   Note: Integer tensors (used for indexing) should not be converted.

        Returns:
            The model moved to the specified device.
        """

        def _is_floating_point(tensor: torch.Tensor) -> bool:
            """Check if tensor is floating point (should have dtype converted)."""
            return tensor.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16)

        # First, move the model parameters
        model = self.model.to(device)
        if dtype is not None:
            model = model.to(dtype)

        # Then handle torchvision-specific cached tensors
        for module in model.modules():
            # Handle AnchorGenerator's cell_anchors cache
            if hasattr(module, "cell_anchors") and module.cell_anchors is not None:
                new_anchors = []
                for anchor in module.cell_anchors:
                    if isinstance(anchor, torch.Tensor):
                        anchor = anchor.to(device)  # noqa: PLW2901
                        if dtype is not None and _is_floating_point(anchor):
                            anchor = anchor.to(dtype)  # noqa: PLW2901
                    new_anchors.append(anchor)
                module.cell_anchors = new_anchors

        return model

    def _customize_inputs(self, entity: SampleBatch) -> dict[str, Any]:
        if isinstance(entity.images, list):
            entity.images, entity.imgs_info = stack_batch(entity.images, entity.imgs_info, pad_size_divisor=32)  # type: ignore[arg-type,assignment]
        return {"entity": entity}

    def _customize_outputs(
        self,
        outputs: dict | list[dict],  # type: ignore[override]
        inputs: SampleBatch,
    ) -> PredictionBatch | BatchLoss:
        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = BatchLoss()
            for loss_name, loss_value in outputs.items():
                if isinstance(loss_value, Tensor):
                    losses[loss_name] = loss_value
                elif isinstance(loss_value, list):
                    losses[loss_name] = sum(_loss.mean() for _loss in loss_value)
            # pop acc from losses
            losses.pop("acc", None)
            return losses

        scores: list[Tensor] = []
        bboxes: list[tv_tensors.BoundingBoxes] = []
        labels: list[torch.LongTensor] = []
        masks: list[tv_tensors.Mask] = []

        # XAI wraps prediction under dictionary with key "predictions"
        predictions = outputs["predictions"] if isinstance(outputs, dict) else outputs
        for img_info, prediction in zip(inputs.imgs_info, predictions):  # type: ignore[arg-type]
            scores.append(prediction["scores"])
            bboxes.append(
                tv_tensors.BoundingBoxes(
                    prediction["boxes"],
                    format="XYXY",
                    canvas_size=img_info.ori_shape,  # type: ignore[union-attr]
                ),
            )
            output_masks = tv_tensors.Mask(
                prediction["masks"],
                dtype=torch.bool,
            )
            masks.append(output_masks)
            labels.append(prediction["labels"])

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

            return PredictionBatch(
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=scores,
                bboxes=bboxes,
                masks=masks,
                labels=labels,
                saliency_map=list(saliency_map),
                feature_vector=list(feature_vector),
            )

        return PredictionBatch(
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            masks=masks,
            labels=labels,
        )

    @property
    def _exporter(self) -> ModelExporter:
        """Creates ModelExporter object that can export the model."""
        return LightningModelExporter(
            task_level_export_parameters=self._export_parameters,
            data_input_params=self.data_input_params,
            resize_mode="fit_to_window",
            pad_value=0,
            swap_rgb=False,
            via_onnx=False,
            onnx_export_configuration={
                "input_names": ["image"],
                "output_names": ["boxes", "labels", "masks"],
                "dynamic_axes": {
                    "image": {0: "batch"},
                    "boxes": {0: "batch", 1: "num_dets"},
                    "labels": {0: "batch", 1: "num_dets"},
                    "masks": {0: "batch", 1: "num_dets", 2: "height", 3: "width"},
                },
                "opset_version": 18,
                "autograd_inlining": False,
                "dynamo": False,
            },
            output_names=["boxes", "labels", "masks", "feature_vector", "saliency_map"] if self.explain_mode else None,
        )

    def forward_for_tracing(self, inputs: Tensor) -> tuple[Tensor, ...]:
        """Forward function for export."""
        shape = (int(inputs.shape[2]), int(inputs.shape[3]))
        meta_info = {
            "image_shape": shape,
        }
        meta_info_list = [meta_info] * len(inputs)
        return self.model.export(
            inputs,
            meta_info_list,
            explain_mode=self.explain_mode,
            with_nms=True,
        )

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for torchvision MaskRCNN."""
        return {
            "ignored_scope": {
                "patterns": [r".*mask_roi_pool.*aten::scatter.*"],
                "validate": False,
            },
        }
