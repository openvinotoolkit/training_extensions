# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""RTDetr model implementations."""

from __future__ import annotations

import copy
import re
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import Tensor, nn
from torchvision.ops import box_convert
from torchvision.tv_tensors import BoundingBoxFormat

from otx.algo.detection.backbones import PResNet
from otx.algo.detection.detectors import DETR
from otx.algo.detection.heads import RTDETRTransformer
from otx.algo.detection.necks import HybridEncoder
from otx.core.config.data import TileConfig
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.detection import DetBatchDataEntity, DetBatchPredEntity
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.metrics.fmeasure import MeanAveragePrecisionFMeasureCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.detection import ExplainableOTXDetModel

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallable
    from otx.core.schedulers import LRSchedulerListCallable
    from otx.core.types.label import LabelInfoTypes


PRETRAINED_ROOT: str = "https://github.com/lyuwenyu/storage/releases/download/v0.1/"

PRETRAINED_WEIGHTS: dict[str, str] = {
    "rtdetr_18": PRETRAINED_ROOT + "rtdetr_r18vd_5x_coco_objects365_from_paddle.pth",
    "rtdetr_50": PRETRAINED_ROOT + "rtdetr_r50vd_2x_coco_objects365_from_paddle.pth",
    "rtdetr_101": PRETRAINED_ROOT + "rtdetr_r101vd_2x_coco_objects365_from_paddle.pth",
}


class RTDETR(ExplainableOTXDetModel):
    """OTX Detection model class for RTDETR.

    Default input size per model:
        - ssd_mobilenetv2 : (640, 640)
    """

    input_size_multiplier = 32
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (255.0, 255.0, 255.0)

    def __init__(
        self,
        model_name: Literal["rtdetr_18", "rtdetr_50", "rtdetr_101"],
        label_info: LabelInfoTypes,
        input_size: tuple[int, int] = (640, 640),
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MeanAveragePrecisionFMeasureCallable,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
    ) -> None:
        self.load_from: str = PRETRAINED_WEIGHTS[model_name]
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

    def _build_model(self, num_classes: int) -> DETR:
        backbone = PResNet(model_name=self.model_name)
        encoder = HybridEncoder(
            model_name=self.model_name,
            eval_spatial_size=self.input_size,
        )
        decoder = RTDETRTransformer(
            model_name=self.model_name,
            num_classes=num_classes,
            eval_spatial_size=self.input_size,
        )

        optimizer_configuration = [
            # no weight decay for norm layers in backbone
            {"params": "^(?=.*backbone)(?=.*norm).*$", "weight_decay": 0.0, "lr": 0.00001},
            # lr for the backbone, but not norm layers is 0.00001
            {"params": "^(?=.*backbone)(?!.*norm).*$", "lr": 0.00001},
            # no weight decay for norm layers and biases in encoder and decoder layers
            {"params": "^(?=.*(?:encoder|decoder))(?=.*(?:norm|bias)).*$", "weight_decay": 0.0},
        ]

        return DETR(
            backbone=backbone,
            encoder=encoder,
            decoder=decoder,
            num_classes=num_classes,
            optimizer_configuration=optimizer_configuration,
            input_size=self.input_size[0],
        )

    def _customize_inputs(
        self,
        entity: DetBatchDataEntity,
        pad_size_divisor: int = 32,
        pad_value: int = 0,
    ) -> dict[str, Any]:
        targets: list[dict[str, Any]] = []
        # prepare bboxes for the model
        for bb, ll in zip(entity.bboxes, entity.labels):
            # convert to cxcywh if needed
            converted_bboxes = (
                box_convert(bb, in_fmt="xyxy", out_fmt="cxcywh") if bb.format == BoundingBoxFormat.XYXY else bb
            )
            # normalize the bboxes
            scaled_bboxes = converted_bboxes / torch.tensor(bb.canvas_size[::-1]).tile(2)[None].to(
                converted_bboxes.device,
            )
            targets.append({"boxes": scaled_bboxes, "labels": ll})

        return {
            "images": entity.images,
            "targets": targets,
        }

    def _customize_outputs(
        self,
        outputs: list[torch.Tensor] | dict,  # type: ignore[override]
        inputs: DetBatchDataEntity,
    ) -> DetBatchPredEntity | OTXBatchLossEntity:
        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            for k, v in outputs.items():
                if isinstance(v, list):
                    losses[k] = sum(v)
                elif isinstance(v, Tensor):
                    losses[k] = v
                else:
                    msg = "Loss output should be list or torch.tensor but got {type(v)}"
                    raise TypeError(msg)
            return losses

        scores, bboxes, labels = self.model.postprocess(outputs, [img_info.img_shape for img_info in inputs.imgs_info])

        return DetBatchPredEntity(
            batch_size=len(outputs),
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
            via_onnx=False,
            onnx_export_configuration={
                "input_names": ["images"],
                "output_names": ["bboxes", "labels", "scores"],
                "dynamic_axes": {
                    "images": {0: "batch"},
                    "boxes": {0: "batch", 1: "num_dets"},
                    "labels": {0: "batch", 1: "num_dets"},
                    "scores": {0: "batch", 1: "num_dets"},
                },
                "autograd_inlining": False,
                "opset_version": 16,
            },
            output_names=["bboxes", "labels", "scores"],
        )

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for RT-DETR."""
        return {"model_type": "transformer"}
