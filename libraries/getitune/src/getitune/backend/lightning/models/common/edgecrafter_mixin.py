# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""EdgeCrafter Mixin providing shared logic for Detection and Instance Segmentation models.

Modified from EdgeCrafter (https://github.com/Intellindust-AI-Lab/EdgeCrafter).
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Protocol, cast

import torch
from torch import Tensor, nn
from torchvision import tv_tensors
from torchvision.ops import box_convert
from torchvision.tv_tensors import BoundingBoxFormat

from getitune.backend.lightning.models.detection.backbones.ecvit import ECViTAdapter
from getitune.backend.lightning.models.detection.detectors.edgecrafter import ECDETRDetector
from getitune.backend.lightning.models.detection.heads.ec_decoder import ECTransformer
from getitune.backend.lightning.models.detection.losses.ec_loss import ECCriterion
from getitune.backend.lightning.models.detection.necks.dfine_hybrid_encoder import HybridEncoder
from getitune.data.entity.base import BatchLoss
from getitune.data.entity.sample import PredictionBatch, SampleBatch

if TYPE_CHECKING:
    from lightning.pytorch.cli import OptimizerCallable

    from getitune.backend.lightning.models.base import DataInputParams


class _EdgeCrafterHost(Protocol):
    """Attributes/behaviour :class:`EdgeCrafterMixin` expects from its host class.

    Concrete host classes (:class:`~getitune.backend.lightning.models.detection.edgecrafter.EdgeCrafter`
    and :class:`~getitune.backend.lightning.models.instance_segmentation.edgecrafter_inst.EdgeCrafterInst`)
    provide these either directly (as instance/class attributes) or by inheriting them from
    :class:`~getitune.backend.lightning.models.base.LightningModel`.
    """

    model: nn.Module
    model_name: str
    data_input_params: DataInputParams
    optimizer_callable: OptimizerCallable
    # NOTE: typed permissively (rather than `LRSchedulerCallable | LRSchedulerListCallable`)
    # because pyrefly infers a narrower concrete type for this read-write attribute on the
    # host classes, and Protocol attribute matching is invariant; `configure_optimizers`
    # below already handles both the single-scheduler and list-of-schedulers cases.
    scheduler_callable: Callable[..., Any]
    multi_scale: bool
    training: bool

    @property
    def explain_mode(self) -> bool:
        """Whether the host model is in explain (XAI) mode; read-only from the Mixin's perspective."""
        ...

    @property
    def num_classes(self) -> int:
        """Number of target classes; read-only from the Mixin's perspective."""
        ...

    # Per-variant backbone/proj_dim/LR config; shared, read-only lookup table.
    _EC_MODEL_CFGS: ClassVar[dict[str, dict[str, Any]]]

    # Per-task configuration, defined as ClassVars on concrete subclasses.
    _loss_weights: ClassVar[dict[str, float]]
    _matcher_cost_dict: ClassVar[dict[str, int | float] | None]
    _mask_downsample_ratio: ClassVar[int | None]
    _backbone_key: ClassVar[str]


class EdgeCrafterMixin:
    """Mixin providing shared EdgeCrafter functionality for detection and instance segmentation.

    Concrete sub-classes must set:

    * ``_loss_weights`` — per-task loss weight dict passed to :class:`ECCriterion`.
    * ``model_name`` — one of ``"edgecrafter_{s,m,l,x}"``.
    * ``data_input_params`` — :class:`DataInputParams` with ``input_size``.
    * ``multi_scale`` — whether multi-scale training is enabled.
    * ``num_classes`` — number of target classes.

    Concrete sub-classes may override (defaults below suit plain detection):

    * ``_matcher_cost_dict`` — Hungarian matcher cost overrides. Defaults to ``None``.
    * ``_mask_downsample_ratio`` — mask head output stride ratio. Defaults to ``None``
      (no segmentation head).
    * ``_backbone_key`` — key into ``_EC_MODEL_CFGS`` selecting the backbone variant.
      Defaults to ``"backbone_name"``.

    The mixin overrides ``_customize_inputs``, ``_customize_outputs``, and
    ``configure_optimizers`` so that both :class:`EdgeCrafter`
    (detection) and :class:`EdgeCrafterInst` (instance segmentation) can share
    the same core logic. The mixin itself stays agnostic to which concrete task
    it is serving — everything task-specific is read from the host class's
    ClassVar configuration (see :class:`_EdgeCrafterHost`).
    """

    _matcher_cost_dict: ClassVar[dict[str, int | float] | None] = None
    _mask_downsample_ratio: ClassVar[int | None] = None
    _backbone_key: ClassVar[str] = "backbone_name"

    # Per-variant backbone, proj_dim, and per-layer LR config.
    _EC_MODEL_CFGS: ClassVar[dict[str, dict[str, Any]]] = {
        "edgecrafter_s": {
            "backbone_name": "ecvitt",
            "seg_backbone_name": "ecseg_vitt",
            "proj_dim": None,
            "backbone_lr": 0.000025,
        },
        "edgecrafter_m": {
            "backbone_name": "ecvittplus",
            "seg_backbone_name": "ecseg_vittplus",
            "proj_dim": None,
            "backbone_lr": 0.000025,
        },
        "edgecrafter_l": {
            "backbone_name": "ecvits",
            "seg_backbone_name": "ecseg_vits",
            "proj_dim": 256,
            "backbone_lr": 0.000005,
        },
        "edgecrafter_x": {
            "backbone_name": "ecvitsplus",
            "seg_backbone_name": "ecseg_vitsplus",
            "proj_dim": 256,
            "backbone_lr": 0.0000025,
        },
    }

    def _build_ec_model(
        self: _EdgeCrafterHost,
        num_classes: int,
        *,
        backbone_lr: float | None = None,
    ) -> ECDETRDetector:
        """Construct the full EdgeCrafter model for detection or instance segmentation.

        Steps:
        1. Build :class:`ECViTAdapter` backbone (det or seg weights variant, selected
           via ``self._backbone_key``).
        2. Build :class:`HybridEncoder` neck.
        3. Build :class:`ECTransformer` decoder (with seg head when
           ``self._mask_downsample_ratio`` is set).
        4. Build :class:`ECCriterion` from ``self._loss_weights`` /
           ``self._matcher_cost_dict``.
        5. Wrap everything in :class:`ECDETRDetector`.

        Args:
            num_classes: Number of target classes.
            backbone_lr: Optional override for the backbone learning rate.
                Defaults to the per-variant value in ``_EC_MODEL_CFGS``.

        Returns:
            Configured :class:`ECDETRDetector` instance.
        """
        cfg = self._EC_MODEL_CFGS[self.model_name]

        if self.data_input_params.input_size is None:
            msg = "input_size must not be None."
            raise ValueError(msg)
        input_size: tuple[int, int] = self.data_input_params.input_size

        backbone = ECViTAdapter(model_name=cfg[self._backbone_key], proj_dim=cfg["proj_dim"])
        encoder = HybridEncoder(model_name=self.model_name)
        decoder = ECTransformer(
            model_name=self.model_name,
            num_classes=num_classes,
            eval_spatial_size=input_size,
            mask_downsample_ratio=self._mask_downsample_ratio,
        )

        criterion = ECCriterion(
            weight_dict=self._loss_weights,
            alpha=0.75,
            gamma=1.5,
            reg_max=32,
            num_classes=num_classes,
            matcher_cost_dict=self._matcher_cost_dict,
        )

        backbone_lr = backbone_lr if backbone_lr is not None else cfg["backbone_lr"]
        optimizer_configuration = [
            {"params": r"^(?=.*backbone)(?!.*(?:norm|bn|bias)).*$", "lr": backbone_lr},
            {"params": r"^(?=.*backbone)(?=.*(?:norm|bn|bias)).*$", "lr": backbone_lr, "weight_decay": 0.0},
            {"params": r"^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$", "weight_decay": 0.0},
        ]

        model = ECDETRDetector(
            backbone=backbone,
            encoder=encoder,
            decoder=decoder,
            criterion=criterion,
            num_classes=num_classes,
            optimizer_configuration=optimizer_configuration,
            multi_scale=self.multi_scale,
            input_size=input_size[0],
        )
        model.init_weights()
        return model

    def _customize_inputs(  # pyrefly: ignore[bad-override]
        self: _EdgeCrafterHost,
        entity: SampleBatch,
    ) -> dict[str, Any]:
        """Convert getitune :class:`SampleBatch` to EdgeCrafter input format.

        Handles detection (boxes + labels) and instance segmentation
        (boxes + labels + masks) depending on what the entity contains.

        Args:
            entity: getitune data batch.

        Returns:
            Dict with ``images`` (Tensor) and ``targets`` (list of per-image dicts).
        """
        targets: list[dict[str, Any]] = []
        has_masks = entity.masks is not None

        if entity.bboxes is not None and entity.labels is not None:
            iterables: tuple
            if has_masks and entity.masks is not None:
                iterables = (entity.bboxes, entity.labels, entity.masks)
            else:
                iterables = (entity.bboxes, entity.labels)

            for items in zip(*iterables):
                bb, ll = items[0], items[1]
                mm = items[2] if has_masks else None

                if bb.numel() > 0 and getattr(bb, "canvas_size", None) is not None:
                    h, w = bb.canvas_size
                    converted = (
                        box_convert(bb, in_fmt="xyxy", out_fmt="cxcywh") if bb.format == BoundingBoxFormat.XYXY else bb
                    )
                    device = converted.device
                    scaled_bboxes = converted / torch.tensor([w, h, w, h], device=device, dtype=torch.float32)
                else:
                    h, w = entity.images.shape[2:]  # pyrefly: ignore[missing-attribute]
                    device = getattr(entity.images, "device", torch.device("cpu"))
                    scaled_bboxes = torch.zeros((0, 4), device=device, dtype=torch.float32)

                target: dict[str, Any] = {
                    "boxes": scaled_bboxes,
                    "labels": ll,
                    "size": torch.tensor([h, w], device=device),
                    "orig_size": torch.tensor([h, w], device=device),
                }
                if mm is not None:
                    target["masks"] = mm

                targets.append(target)

        if self.explain_mode:
            return {"entity": entity}

        return {
            "images": entity.images,
            "targets": targets,
        }

    def _customize_outputs(  # pyrefly: ignore[bad-override]
        self: _EdgeCrafterHost,
        outputs: dict[str, Any] | tuple | list,
        inputs: SampleBatch,
    ) -> PredictionBatch | BatchLoss:
        """Convert model outputs to getitune format.

        During training, wraps the loss dict in a :class:`BatchLoss`.
        During inference, calls ``self.model.postprocess`` and formats predictions.
        When masks are returned (instance segmentation), they are included in the
        :class:`PredictionBatch`.

        Args:
            outputs: Loss dict (training) or raw decoder output dict (inference).
            inputs: Original getitune data batch.

        Returns:
            :class:`BatchLoss` during training, :class:`PredictionBatch` during inference.
        """
        if self.training:
            if not isinstance(outputs, dict):
                msg = f"Expected dict during training, got {type(outputs)}"
                raise TypeError(msg)

            losses = BatchLoss()
            for k, v in outputs.items():
                if isinstance(v, Tensor):
                    losses[k] = v
                elif isinstance(v, list):
                    losses[k] = sum(
                        (_loss.mean() for _loss in v if isinstance(_loss, Tensor)),
                        torch.tensor(0.0),
                    )  # pyrefly: ignore[unsupported-operation]
                # nested dicts (e.g. dn_meta) are silently skipped
            return losses

        original_sizes = [img_info.ori_shape for img_info in inputs.imgs_info]  # type: ignore[union-attr]
        model = cast("ECDETRDetector", self.model)
        result = model.postprocess(cast("dict[str, Tensor]", outputs), original_sizes)

        prediction_kwargs: dict[str, Any] = {
            "images": inputs.images,
            "imgs_info": inputs.imgs_info,
        }

        if len(result) == 4:  # detection + masks (instance seg)
            scores_list, boxes_list, labels_list, masks_list = result
            prediction_kwargs["scores"] = scores_list
            prediction_kwargs["bboxes"] = boxes_list
            prediction_kwargs["labels"] = labels_list

            formatted_masks = []
            for masks, img_info in zip(masks_list, inputs.imgs_info):  # type: ignore[union-attr, arg-type]
                if masks is not None and masks.numel() > 0:
                    formatted_masks.append(tv_tensors.Mask(masks, dtype=torch.uint8))  # type: ignore[call-overload]
                else:
                    formatted_masks.append(
                        tv_tensors.Mask(  # type: ignore[call-overload]
                            torch.zeros(
                                (0, img_info.img_shape[0], img_info.img_shape[1]),  # type: ignore[union-attr]
                                dtype=torch.bool,
                            ),
                        )
                    )
            prediction_kwargs["masks"] = formatted_masks
        else:  # detection only
            scores_list, boxes_list, labels_list = result
            prediction_kwargs["scores"] = scores_list
            prediction_kwargs["bboxes"] = boxes_list
            prediction_kwargs["labels"] = labels_list

        return PredictionBatch(**prediction_kwargs)

    def configure_optimizers(  # pyrefly: ignore[bad-override]
        self: _EdgeCrafterHost,
    ) -> tuple[list[torch.optim.Optimizer], list[dict[str, Any]]]:
        """Configure optimizer and learning-rate schedulers.

        Uses :meth:`RTDETR._get_optim_params` to build per-param-group
        configurations from the regex patterns in
        ``self.model.optimizer_configuration``.

        Returns:
            Two lists: optimizer list and lr-scheduler config list.
        """
        from getitune.backend.lightning.models.detection.rtdetr import RTDETR

        param_groups = RTDETR._get_optim_params(  # noqa: SLF001
            cast("ECDETRDetector", self.model).optimizer_configuration,
            self.model,
        )
        optimizer = self.optimizer_callable(param_groups)
        schedulers = self.scheduler_callable(optimizer)

        def _ensure_list(item: Any) -> list:  # noqa: ANN401
            return item if isinstance(item, list) else [item]

        lr_scheduler_configs = []
        for scheduler in _ensure_list(schedulers):
            cfg: dict[str, Any] = {"scheduler": scheduler}
            if hasattr(scheduler, "interval"):
                cfg["interval"] = scheduler.interval
            if hasattr(scheduler, "monitor"):
                cfg["monitor"] = scheduler.monitor
            lr_scheduler_configs.append(cfg)

        return [optimizer], lr_scheduler_configs

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for EdgeCrafter."""
        return {"model_type": "transformer"}

    # ------------------------------------------------------------------
    # Helpers used by sub-classes
    # ------------------------------------------------------------------

    @staticmethod
    def _make_is_export_prediction(
        _inputs: SampleBatch | Tensor,
        result: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Reformat deploy-mode dict for MaskRCNN-compatible ONNX export.

        Merges ``bboxes`` + ``scores`` into a ``[B, Q, 5]`` tensor,
        scales bboxes from normalised to pixel coordinates, and returns
        the three ONNX output tensors: ``boxes``, ``labels``, ``masks``.

        Args:
            inputs: Original image batch (used for spatial dimensions).
            result: Deploy-mode postprocess output dict.

        Returns:
            Dict with ``boxes`` [B, Q, 5], ``labels`` [B, Q], ``masks`` [B, Q, H, W].
        """
        bboxes = result["bboxes"]  # [B, Q, 4] - already pixel coords
        scores = result["scores"]  # [B, Q]
        labels = result["labels"]  # [B, Q]
        masks = result.get("masks")  # [B, Q, H, W] or None

        boxes_with_scores = torch.cat([bboxes, scores.unsqueeze(-1)], dim=-1)  # [B, Q, 5]
        out: dict[str, Tensor] = {"boxes": boxes_with_scores, "labels": labels}
        if masks is not None:
            out["masks"] = masks
        return out

    @staticmethod
    def _default_is_meta(inputs: Tensor) -> list[dict[str, Any]]:
        """Build a minimal meta-info list for export tracing."""
        shape = (int(inputs.shape[2]), int(inputs.shape[3]))
        meta = {
            "pad_shape": shape,
            "batch_input_shape": shape,
            "img_shape": shape,
            "scale_factor": (1.0, 1.0),
        }
        return [copy.copy(meta) for _ in range(inputs.shape[0])]
