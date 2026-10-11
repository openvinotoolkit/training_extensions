# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RF-DETR Mixin for shared logic between Detection and Instance Segmentation models.

This mixin encapsulates common functionality used by both RFDETR (detection) and
RFDETRInst (instance segmentation) models.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch
from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import TrainConfig
from rfdetr.models import build_criterion_from_config, build_model_from_config, load_pretrain_weights
from rfdetr.models._defaults import MODEL_DEFAULTS
from rfdetr.models.backbone import Joiner
from torch.hub import download_url_to_file
from torchvision import tv_tensors
from torchvision.ops import box_convert

from getitune.backend.lightning.models.detection.detectors.rfdetr import RFDETRDetector
from getitune.backend.lightning.models.detection.utils import limit_batch_objects
from getitune.data.entity.base import BatchLoss
from getitune.data.entity.sample import PredictionBatch, SampleBatch
from getitune.types.export import ExportFormat
from getitune.types.precision import Precision

if TYPE_CHECKING:
    from types import SimpleNamespace

    from rfdetr.models.backbone import Backbone
    from rfdetr.models.lwdetr import LWDETR

    from getitune.types import PathLike


logger = logging.getLogger(__name__)


def _get_param_dict(args: SimpleNamespace, model_without_ddp: torch.nn.Module) -> list[dict[str, Any]]:
    if not isinstance(model_without_ddp.backbone, Joiner):
        msg = f"Expected backbone to be Joiner, got {type(model_without_ddp.backbone).__name__}"
        raise TypeError(msg)

    backbone = cast("Backbone", model_without_ddp.backbone[0])
    backbone_named_param_lr_pairs = backbone.get_named_param_lr_pairs(args, prefix="backbone.0")
    backbone_param_lr_pairs = [param_dict for _, param_dict in backbone_named_param_lr_pairs.items()]

    decoder_key = "transformer.decoder"
    decoder_params = [p for n, p in model_without_ddp.named_parameters() if decoder_key in n and p.requires_grad]
    decoder_param_lr_pairs = [{"params": param, "lr": args.lr * args.lr_component_decay} for param in decoder_params]

    other_params = [
        p
        for n, p in model_without_ddp.named_parameters()
        if (n not in backbone_named_param_lr_pairs and decoder_key not in n and p.requires_grad)
    ]
    other_param_dicts = [{"params": param, "lr": args.lr} for param in other_params]

    return other_param_dicts + backbone_param_lr_pairs + decoder_param_lr_pairs


class RFDETRMixin:
    """Mixin class providing shared RF-DETR functionality for detection and instance segmentation."""

    pretrained_urls: ClassVar[dict[str, str]]
    _model_config_mapping: ClassVar[dict[str, type]]

    def _build_rfdetr_model(
        self,
        num_classes: int,
        *,
        gradient_checkpointing: bool = False,
    ) -> RFDETRDetector:
        """Build the core RF-DETR model.

        Handles the full pipeline shared by both detection and instance segmentation:
        1. Build LWDETR from per-variant config.
        2. Build criterion and postprocessor.
        3. Wrap in ``RFDETRDetector``.

        Args:
            num_classes: Number of target classes.
            gradient_checkpointing: Whether to enable gradient checkpointing.

        Returns:
            Configured ``RFDETRDetector`` instance.
        """
        model_config_cls = self._model_config_mapping[self.model_name]  # type: ignore[attr-defined]

        model_config = model_config_cls()
        model_config.num_classes = num_classes
        model_config.gradient_checkpointing = gradient_checkpointing
        train_config = TrainConfig(dataset_dir=".")

        lwdetr: LWDETR = build_model_from_config(model_config, train_config)

        self._model_config = model_config

        criterion, postprocessor = build_criterion_from_config(model_config, train_config)

        self.rfdetr_args = _namespace_from_configs(  # type: ignore[attr-defined]
            model_config, train_config, MODEL_DEFAULTS
        )

        return RFDETRDetector(
            lwdetr_model=lwdetr,
            criterion=criterion,
            postprocessor=postprocessor,
            rfdetr_args=self.rfdetr_args,  # type: ignore[attr-defined]
            input_size=self.data_input_params.input_size[0],  # type: ignore[attr-defined]
            multi_scale=self.multi_scale,  # type: ignore[attr-defined]
        )

    def _customize_inputs(  # pyrefly: ignore[bad-override]
        self,
        entity: SampleBatch,
    ) -> dict[str, Any]:
        """Convert getitune batch format to RF-DETR input format.

        Handles both detection (boxes + labels) and instance segmentation
        (boxes + labels + masks) depending on what the entity contains.

        Args:
            entity: getitune data batch.

        Returns:
            Dict with 'images' and 'targets' for the model.
        """
        targets: list[dict[str, Any]] = []
        has_masks = entity.masks is not None

        required_attrs = ("bboxes", "labels", "masks") if has_masks else ("bboxes", "labels")
        if all(getattr(entity, attr) is not None for attr in required_attrs):
            iterables: tuple = (
                (entity.bboxes, entity.labels, entity.masks) if has_masks else (entity.bboxes, entity.labels)  # type: ignore[assignment]
            )
            for _i, items in enumerate(zip(*iterables)):
                bb, ll = items[0], items[1]
                mm = items[2] if has_masks else None

                if bb.numel() > 0 and getattr(bb, "canvas_size", None) is not None:
                    h, w = bb.canvas_size
                    boxes_cxcywh = box_convert(bb.data, in_fmt="xyxy", out_fmt="cxcywh")
                    device = boxes_cxcywh.device
                    scaled_bboxes = boxes_cxcywh / torch.tensor(
                        [w, h, w, h],
                        device=device,
                        dtype=torch.float32,
                    )
                else:
                    h, w = entity.images.shape[2:]  # pyrefly: ignore[missing-attribute]
                    device = getattr(entity.images, "device", torch.device("cpu"))
                    scaled_bboxes = torch.zeros((0, 4), device=device, dtype=torch.float32)

                # Defensive safeguard: the RF-DETR criterion (and the Hungarian
                # matcher) concatenate boxes, labels and masks per target and
                # require their per-image counts to match. The data pipeline
                # should already guarantee this, but geometric transforms
                # (cropping, tiling, box sanitization) can occasionally drop one
                # annotation type without the others. If counts ever diverge we
                # align boxes/labels/masks to their common length rather than
                # crash deep inside the matcher. This runs unconditionally so it
                # also protects the detection path (boxes vs labels), not only
                # instance segmentation (boxes vs labels vs masks).
                n_boxes = int(scaled_bboxes.shape[0])
                n_labels = int(ll.shape[0])
                n_masks = int(mm.shape[0]) if mm is not None else n_boxes
                if not (n_boxes == n_labels == n_masks):
                    n = min(n_boxes, n_labels, n_masks)
                    logger.warning(
                        "RF-DETR target has mismatched annotation counts "
                        "(boxes=%d, labels=%d, masks=%d); aligning to %d.",
                        n_boxes,
                        n_labels,
                        n_masks,
                        n,
                    )
                    ll = ll[:n]
                    if mm is not None:
                        # Instance segmentation: rebuild the boxes from the
                        # (trimmed) masks so box[i] is the tight box of mask[i].
                        # The RF-DETR matcher pairs per-target box and mask costs,
                        # so preserving this correspondence is essential — a plain
                        # trim could pair a box with a different instance's mask.
                        mm = mm[:n]
                        if n > 0:
                            from getitune.data.utils.structures.mask.mask_target import masks_to_boxes

                            xyxy = masks_to_boxes(mm, dtype=torch.float32).to(device)
                            scaled_bboxes = box_convert(xyxy, in_fmt="xyxy", out_fmt="cxcywh") / torch.tensor(
                                [w, h, w, h],
                                device=device,
                                dtype=torch.float32,
                            )
                        else:
                            scaled_bboxes = torch.zeros((0, 4), device=device, dtype=torch.float32)
                    else:
                        scaled_bboxes = scaled_bboxes[:n]

                target: dict[str, Any] = {
                    "boxes": scaled_bboxes,
                    "labels": ll,
                    "size": torch.tensor([h, w], device=device),
                    "orig_size": torch.tensor([h, w], device=device),
                }
                if mm is not None:
                    target["masks"] = mm

                targets.append(target)

        # Apply batch-level object limiting if configured (only during training)
        if self.training and self.max_total_objects_per_batch is not None:  # type: ignore[attr-defined]
            targets = limit_batch_objects(targets, self.max_total_objects_per_batch)  # type: ignore[attr-defined]

        if self.explain_mode:  # type: ignore[attr-defined]
            return {"entity": entity}

        return {
            "images": entity.images,
            "targets": targets,
        }

    def _customize_outputs(  # pyrefly: ignore[bad-override]
        self,
        outputs: tuple[torch.Tensor, ...] | dict[str, Any],
        inputs: SampleBatch,
    ) -> PredictionBatch | BatchLoss:
        """Convert model outputs to getitune format.

        Handles both detection and instance segmentation outputs.
        When the postprocessor returns masks (4th element is not ``None``),
        they are formatted and included in the prediction batch.

        Args:
            outputs: Model outputs (loss dict during training, predictions during inference).
            inputs: Original getitune data batch.

        Returns:
            ``PredictionBatch`` during inference, ``BatchLoss`` during training.
        """
        if self.training:  # type: ignore[attr-defined]
            if not isinstance(outputs, dict):
                msg = f"Expected dict during training, got {type(outputs)}"
                raise TypeError(msg)

            losses = BatchLoss()
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor):
                    losses[k] = v
                elif isinstance(v, list):
                    losses[k] = sum(
                        (_loss.mean() for _loss in v),
                        torch.tensor(0.0),
                    )  # pyrefly: ignore[unsupported-operation]
                else:
                    msg = f"Loss output should be list or torch.Tensor but got {type(v)}"
                    raise TypeError(msg)
            return losses

        image_shapes = [img_info.ori_shape for img_info in inputs.imgs_info]  # type: ignore[union-attr]
        scores_list, boxes_list, labels_list, masks_list = self.model.postprocess(  # type: ignore[attr-defined]  # pyrefly: ignore[not-callable]
            outputs,
            image_shapes,
        )

        prediction_kwargs: dict[str, Any] = {
            "images": inputs.images,
            "imgs_info": inputs.imgs_info,
            "scores": scores_list,
            "bboxes": boxes_list,
            "labels": labels_list,
        }

        # Include masks when the postprocessor produces them (instance segmentation)
        if masks_list is not None and any(m is not None for m in masks_list):
            formatted_masks = []
            for masks, img_info in zip(masks_list, inputs.imgs_info):  # type: ignore[union-attr, arg-type]
                if masks is not None and len(masks) > 0:
                    formatted_masks.append(
                        tv_tensors.Mask(masks, dtype=torch.uint8),  # type: ignore[call-overload]
                    )
                else:
                    formatted_masks.append(
                        tv_tensors.Mask(
                            torch.zeros(
                                (0, img_info.img_shape[0], img_info.img_shape[1]),  # type: ignore[union-attr]
                                dtype=torch.bool,
                            ),  # type: ignore[call-overload]
                        ),
                    )
            prediction_kwargs["masks"] = formatted_masks

        if self.explain_mode:  # type: ignore[attr-defined]
            msg = "Explain mode is not supported for RF-DETR model."
            raise ValueError(msg)

        return PredictionBatch(**prediction_kwargs)

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict[str, Any]]]:
        """Configure optimizer and learning-rate schedulers.

        Uses rfdetr's ``get_param_dict`` to create proper parameter groups with
        correct lr and weight_decay settings from rfdetr_args.

        Returns:
            Two lists: optimizer list and lr scheduler config list.
        """
        # Extract default lr and weight_decay from optimizer callable
        dummy = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
        dummy_param_groups = [{"params": [dummy]}]
        default_lr = self.optimizer_callable(dummy_param_groups).param_groups[0]["lr"]  # type: ignore[attr-defined]
        default_weight_decay = self.optimizer_callable(dummy_param_groups).param_groups[0]["weight_decay"]  # type: ignore[attr-defined]

        # Get parameter groups from rfdetr with correct args
        self.rfdetr_args.lr = default_lr  # type: ignore[attr-defined]
        self.rfdetr_args.weight_decay = default_weight_decay  # type: ignore[attr-defined]
        param_groups = _get_param_dict(self.rfdetr_args, self.model.lwdetr)  # type: ignore[attr-defined]  # pyrefly: ignore[bad-argument-type]

        # Create optimizer and schedulers
        optimizer = self.optimizer_callable(param_groups)  # type: ignore[attr-defined]
        schedulers = self.scheduler_callable(optimizer)  # type: ignore[attr-defined]

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

    def forward_for_tracing(self, inputs: torch.Tensor) -> dict[str, Any] | tuple[torch.Tensor, ...]:
        """Forward pass used for model tracing/export."""
        return self.model.export(inputs)  # type: ignore[attr-defined]  # pyrefly: ignore[not-callable]

    @staticmethod
    def _restore_forward_methods(module: torch.nn.Module) -> None:
        """Undo ``module.export()`` forward monkey-patching on all sub-modules.

        Some rfdetr sub-modules (e.g. ``TransformerDecoder``, ``MSDeformAttn``)
        set ``_export = True`` in their ``export()`` method but do **not** swap
        ``forward`` (i.e. they have no ``_forward_origin``).  They instead
        branch on ``self._export`` inside their existing ``forward``.  We must
        reset the flag on *every* module that carries it, not only those whose
        ``forward`` was monkey-patched.
        """
        for m in module.modules():
            if not getattr(m, "_export", False):
                continue
            if hasattr(m, "_forward_origin"):
                m.forward = m._forward_origin  # noqa: SLF001  # pyrefly: ignore[bad-assignment]
            m._export = False  # noqa: SLF001  # pyrefly: ignore[bad-argument-type]

    def export(
        self,
        output_dir: Path,
        base_name: str,
        export_format: ExportFormat,
        precision: Precision = Precision.FP32,
    ) -> Path:
        """Export the model to the requested format."""
        if self.explain_mode:  # pyrefly: ignore[missing-attribute]
            msg = "Explain mode is not supported for RF-DETR model."
            raise ValueError(msg)
        lwdetr = self.model.lwdetr  # pyrefly: ignore[missing-attribute]
        lwdetr.export()  # pyrefly: ignore[missing-attribute]
        try:
            return super().export(output_dir, base_name, export_format, precision)  # type: ignore[misc]
        except Exception:
            self._restore_forward_methods(lwdetr)
            raise
        finally:
            self._restore_forward_methods(lwdetr)

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for RF-DETR."""
        return {"model_type": "transformer"}

    def load_pretrained(self, weights: PathLike | None = None) -> None:
        """Load pretrained weights into the model.

        Load & align pretrained weights (handles head resize internally).
        Reinit classification biases to zero (sigmoid=0.5 neutral prior).

        Args:
            weights (PathLike | None): Path to the pretrained weights file. If None, uses default weights.
        """
        if weights is None:
            pretrain_url = self.pretrained_urls.get(self.model_name)  # type: ignore[attr-defined]
            if pretrain_url:
                cache_dir = Path(os.environ["PRETRAINED_WEIGHTS_CACHE_DIR"])
                cache_dir.mkdir(parents=True, exist_ok=True)
                local_path = cache_dir / Path(pretrain_url).name
                is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
                rank = torch.distributed.get_rank() if is_distributed else 0
                if rank == 0 and not local_path.exists():
                    download_url_to_file(pretrain_url, str(local_path), progress=os.isatty(0))
                if is_distributed:
                    torch.distributed.barrier()
                self._model_config.pretrain_weights = str(local_path)
            else:
                self._model_config.pretrain_weights = None
        else:
            self._model_config.pretrain_weights = str(weights)

        lwdetr = self.model.lwdetr  # pyrefly: ignore[missing-attribute]
        load_pretrain_weights(lwdetr, self._model_config)

        torch.nn.init.zeros_(lwdetr.class_embed.bias)
        if lwdetr.two_stage:
            for enc_cls_embed in lwdetr.transformer.enc_out_class_embed:
                torch.nn.init.zeros_(enc_cls_embed.bias)
