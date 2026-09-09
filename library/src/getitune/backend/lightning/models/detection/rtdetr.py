# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RTDetr model implementations."""

from __future__ import annotations

import copy
import re
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch
from torch import Tensor, nn
from torch.export import Dim
from torchvision.ops import box_convert
from torchvision.tv_tensors import BoundingBoxFormat

from getitune.backend.lightning.exporter.base import ModelExporter
from getitune.backend.lightning.exporter.native import LightningModelExporter
from getitune.backend.lightning.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from getitune.backend.lightning.models.common.target_utils import align_sample_batch_annotations
from getitune.backend.lightning.models.detection.backbones import PResNet
from getitune.backend.lightning.models.detection.base import LightningDetectionModel
from getitune.backend.lightning.models.detection.detectors import DETR
from getitune.backend.lightning.models.detection.heads import RTDETRTransformer
from getitune.backend.lightning.models.detection.necks import HybridEncoder
from getitune.config.data import TileConfig
from getitune.data.entity.base import BatchLoss
from getitune.data.entity.sample import PredictionBatch, SampleBatch
from getitune.metrics.fmeasure import MeanAveragePrecisionFMeasureCallable
from getitune.types.export import TaskLevelExportParameters

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from getitune.backend.lightning.schedulers import LRSchedulerListCallable
    from getitune.metrics import MetricCallable
    from getitune.types import PathLike
    from getitune.types.label import LabelInfoTypes


class RTDETR(LightningDetectionModel):
    """getitune Detection model class for RTDETR.

    Attributes:
        pretrained_urls (ClassVar[dict[str, str]]): Dictionary containing URLs for pretrained weights.
        input_size_multiplier (int): Multiplier for the input size.

    Args:
        label_info (LabelInfoTypes): Information about the labels.
        data_input_params (DataInputParams | dict | None, optional): Parameters for the image data preprocessing.
            If None, uses _default_preprocessing_params.
        model_name (literal, optional): Name of the model to use. Defaults to "rtdetr_50".
        optimizer (OptimizerCallable, optional): Callable for the optimizer. Defaults to DefaultOptimizerCallable.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): Callable for the learning rate scheduler.
            Defaults to DefaultSchedulerCallable.
        metric (MetricCallable, optional): Callable for the metric. Defaults to MeanAveragePrecisionFMeasureCallable.
        multi_scale (bool, optional): Whether to use multi-scale training. Defaults to False.
        torch_compile (bool, optional): Whether to use torch compile. Defaults to False.
        tile_config (TileConfig, optional): Configuration for tiling. Defaults to TileConfig(enable_tiler=False).
        pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
        pretrained_weights (PathLike | None, optional): Path to the pretrained weights file. When None is passed,
            the default pretrained weights will be utilized for fine-tuning. Defaults to None.
    """

    pretrained_urls: ClassVar[dict[str, str]] = {
        "rtdetr_18": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth",
        "rtdetr_50": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth",
        "rtdetr_101": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_2x_coco_objects365_from_paddle.pth",
    }

    input_size_multiplier = 32

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams | dict | None = None,
        model_name: Literal["rtdetr_18", "rtdetr_50", "rtdetr_101"] = "rtdetr_50",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MeanAveragePrecisionFMeasureCallable,
        multi_scale: bool = False,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
        pretrained: bool = True,
        pretrained_weights: PathLike | None = None,
        export_nms: bool = False,
    ) -> None:
        self.multi_scale = multi_scale
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

    def _create_model(self, num_classes: int | None = None) -> DETR:
        num_classes = num_classes if num_classes is not None else self.num_classes
        backbone = PResNet(model_name=self.model_name)
        encoder = HybridEncoder(
            model_name=self.model_name,
            eval_spatial_size=self.data_input_params.input_size,
        )
        decoder = RTDETRTransformer(
            model_name=self.model_name,
            num_classes=num_classes,
            eval_spatial_size=self.data_input_params.input_size,
        )

        optimizer_configuration = [
            # no weight decay for norm layers in backbone
            {"params": "^(?=.*backbone)(?=.*norm).*$", "weight_decay": 0.0, "lr": 0.00001},
            # lr for the backbone, but not norm layers is 0.00001
            {"params": "^(?=.*backbone)(?!.*norm).*$", "lr": 0.00001},
            # no weight decay for norm layers and biases in encoder and decoder layers
            {"params": "^(?=.*(?:encoder|decoder))(?=.*(?:norm|bias)).*$", "weight_decay": 0.0},
        ]

        if self.data_input_params.input_size is None:
            msg = "input_size should not be None."
            raise ValueError(msg)
        model = DETR(
            multi_scale=self.multi_scale,
            backbone=backbone,
            encoder=encoder,
            decoder=decoder,
            num_classes=num_classes,
            optimizer_configuration=optimizer_configuration,
            input_size=self.data_input_params.input_size[0],
        )
        model.init_weights()

        return model

    def _customize_inputs(
        self,
        entity: SampleBatch,
        pad_size_divisor: int = 32,
        pad_value: int = 0,
    ) -> dict[str, Any]:
        targets: list[dict[str, Any]] = []
        # Defensively realign per-image annotation counts so a boxes/labels
        # divergence (e.g. from tiling or crop augmentations) does not crash the
        # Hungarian matcher, which concatenates boxes and labels per target.
        if self.training:
            align_sample_batch_annotations(entity)
        # prepare bboxes for the model
        if entity.bboxes is not None and entity.labels is not None:
            for bb, ll in zip(entity.bboxes, entity.labels):
                # convert to cxcywh if needed
                scaled_bboxes = bb
                if len(bb):
                    converted_bboxes = (
                        box_convert(bb, in_fmt="xyxy", out_fmt="cxcywh") if bb.format == BoundingBoxFormat.XYXY else bb
                    )
                    # normalize the bboxes
                    scaled_bboxes = converted_bboxes / torch.tensor(bb.canvas_size[::-1]).tile(2)[None].to(
                        converted_bboxes.device,
                    )
                h, w = bb.canvas_size
                device = scaled_bboxes.device
                targets.append(
                    {
                        "boxes": scaled_bboxes,
                        "labels": ll,
                        "size": torch.tensor([h, w], device=device),
                        "orig_size": torch.tensor([h, w], device=device),
                    }
                )

        if self.explain_mode:
            return {"entity": entity}

        return {
            "images": entity.images,
            "targets": targets,
        }

    def _customize_outputs(
        self,
        outputs: list[torch.Tensor] | dict,  # type: ignore[override]
        inputs: SampleBatch,
    ) -> PredictionBatch | BatchLoss:
        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = BatchLoss()
            for k, v in outputs.items():
                if isinstance(v, list):
                    losses[k] = sum(v)
                elif isinstance(v, Tensor):
                    losses[k] = v
                else:
                    msg = "Loss output should be list or torch.tensor but got {type(v)}"
                    raise TypeError(msg)
            return losses

        original_sizes = [img_info.ori_shape for img_info in inputs.imgs_info]  # type: ignore[union-attr]
        scores, bboxes, labels = self.model.postprocess(outputs, original_sizes)

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

            return PredictionBatch(
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=scores,
                bboxes=bboxes,
                labels=labels,
                feature_vector=[feature_vector.unsqueeze(0) for feature_vector in outputs["feature_vector"]],
                saliency_map=[saliency_map.to(torch.float32) for saliency_map in outputs["saliency_map"]],
            )

        return PredictionBatch(
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            labels=labels,
        )

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict[str, Any]]]:
        """Configure an optimizer and learning-rate schedulers.

        Configure an optimizer and learning-rate schedulers
        from the given optimizer and scheduler or scheduler list callable in the constructor.
        Generally, there is two lr schedulers. One is for a linear warmup scheduler and
        the other is the main scheduler working after the warmup period.

        Returns:
            Two list. The former is a list that contains an optimizer
            The latter is a list of lr scheduler configs which has a dictionary format.
        """
        param_groups = self._get_optim_params(self.model.optimizer_configuration, self.model)
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

    @staticmethod
    def _get_optim_params(cfg: list[dict[str, Any]] | None, model: nn.Module) -> list[dict[str, Any]]:
        """Perform no bias decay and learning rate correction for the modules.

        The configuration dict should consist of regular expression pattern for the model parameters with "params" key.
        Other optimizer parameters can be added as well.

        E.g.:
            cfg = [{"params": "^((?!b).)*$", "lr": 0.01, "weight_decay": 0.0}, ..]
            The above configuration is for the parameters that do not contain "b".

            ^(?=.*a)(?=.*b).*$         means including a and b
            ^((?!b.)*a((?!b).)*$       means including a but not b
            ^((?!b|c).)*a((?!b|c).)*$  means including a but not (b | c)
        """
        if cfg is None:
            return model.parameters()

        cfg = copy.deepcopy(cfg)

        param_groups = []
        visited = []
        for pg in cfg:
            if "params" not in pg:
                msg = f"The 'params' key should be included in the configuration, but got {pg.keys()}"
                raise ValueError(msg)
            pattern = pg["params"]
            params = {k: v for k, v in model.named_parameters() if v.requires_grad and len(re.findall(pattern, k)) > 0}
            pg["params"] = params.values()
            param_groups.append(pg)
            visited.extend(list(params.keys()))

        names = [k for k, v in model.named_parameters() if v.requires_grad]

        if len(visited) < len(names):
            unseen = set(names) - set(visited)
            params = {k: v for k, v in model.named_parameters() if v.requires_grad and k in unseen}
            param_groups.append({"params": params.values()})
            visited.extend(list(params.keys()))

        return param_groups

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Defines parameters required to export a particular model implementation."""
        # DETR models use Hungarian matching for one-to-one predictions, but on small datasets
        # near-duplicate boxes can still appear. Use a conservative IoU threshold (0.8) to only
        # suppress almost-identical duplicates without removing valid overlapping detections.
        return super()._export_parameters.wrap(iou_threshold=0.8, nms_execute=True)

    @property
    def _exporter(self) -> ModelExporter:
        """Creates ModelExporter object that can export the model."""
        return LightningModelExporter(
            task_level_export_parameters=self._export_parameters,
            data_input_params=self.data_input_params,
            resize_mode="standard",
            swap_rgb=False,
            via_onnx=False,
            onnx_export_configuration={
                "input_names": ["images"],
                "output_names": ["bboxes", "labels", "scores"],
                "dynamic_shapes": {"inputs": {0: Dim("batch")}},
                "autograd_inlining": False,
                "opset_version": 18,
            },
            output_names=["bboxes", "labels", "scores"],
        )

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for RT-DETR."""
        return {"model_type": "transformer"}

    @staticmethod
    def _forward_explain_detection(
        self,  # noqa: ANN001, PLW0211
        entity: SampleBatch,
        mode: str = "tensor",  # noqa: ARG004
    ) -> dict[str, torch.Tensor]:
        """Forward function for explainable detection model."""
        backbone_feats = self.encoder(self.backbone(entity.images))
        predictions = self.decoder(backbone_feats, explain_mode=True)

        raw_logits = DETR.split_and_reshape_logits(
            backbone_feats,
            predictions["raw_logits"],
        )

        saliency_map = self.explain_fn(raw_logits)
        feature_vector = self.feature_vector_fn(backbone_feats)
        predictions.update(
            {
                "feature_vector": feature_vector,
                "saliency_map": saliency_map,
            },
        )

        return predictions
