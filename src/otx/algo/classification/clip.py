# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Hugging-Face pretrained model for the OTX classification."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import nn
from transformers import CLIPModel, CLIPProcessor
from transformers.configuration_utils import PretrainedConfig

from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.classification import (
    MulticlassClsBatchDataEntity,
    MulticlassClsBatchPredEntity,
)
from otx.core.metrics.accuracy import MultiClassClsMetricCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.classification import OTXMulticlassClsModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.label import LabelInfoTypes

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from transformers.models.clip.modeling_clip import CLIPOutput

    from otx.core.metrics import MetricCallable


DEFAULT_INPUT_SIZE = (224, 224)
logger = logging.getLogger(__name__)

CLIP_TYPE = Literal[
    "openai/clip-vit-base-patch32",
]


class CLIP(OTXMulticlassClsModel):
    def __init__(
        self,
        model_name_or_path: CLIP_TYPE,  # https://huggingface.co/models?pipeline_tag=image-classification
        label_info: LabelInfoTypes,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiClassClsMetricCallable,
        torch_compile: bool = False,
        input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
    ) -> None:
        self.model_name = model_name_or_path

        super().__init__(
            label_info=label_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            input_size=input_size,
        )
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.criterion = nn.CrossEntropyLoss()

    def _create_model(self) -> nn.Module:
        model_config, _ = PretrainedConfig.get_config_dict(self.model_name)
        kwargs = {}
        if "image_size" in model_config:
            kwargs["image_size"] = self.input_size[0]
        elif self.input_size != DEFAULT_INPUT_SIZE:
            msg = "There is no 'image_size' argument in the model configuration. There may be unexpected results."
            logger.warning(msg)

        return CLIPModel.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            num_labels=self.label_info.num_classes,
            ignore_mismatched_sizes=True,
            **kwargs,
        )

    def _customize_inputs(self, inputs: MulticlassClsBatchDataEntity) -> dict[str, Any]:
        # TODO: Need to implement for image, text pair dataset
        # inputs.labels to be converted to text mapped with self.label_info idx
        text_prom = [f"{self.label_info.label_names[idx]}" for idx in inputs.labels]
        captions = self.processor(
            text=text_prom,
            images=None,
            return_tensors="pt",
            padding=True,
        )
        input_ids, attention_mask = captions["input_ids"].to(self.device), captions["attention_mask"].to(self.device)

        return {
            "input_ids": input_ids,
            "pixel_values": inputs.images,
            "attention_mask": attention_mask,
            "return_loss": self.training,
        }

    def _customize_outputs(
        self,
        outputs: CLIPOutput,
        inputs: MulticlassClsBatchDataEntity,
    ) -> MulticlassClsBatchPredEntity | OTXBatchLossEntity:
        if self.training:
            return OTXBatchLossEntity(loss=outputs.loss)

        logits_per_image = outputs.logits_per_image
        # logits_per_text = outputs.logits_per_text
        scores = logits_per_image.softmax(dim=1)
        # scores = torch.unbind(logits_per_image, 0)
        preds = logits_per_image.argmax(-1, keepdim=True).unbind(0)

        return MulticlassClsBatchPredEntity(
            batch_size=inputs.batch_size,
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            labels=preds,
        )

    # def forward_for_tracing(self, image: Tensor) -> Tensor | dict[str, Tensor]:
    #     """Model forward function used for the model tracing during model exportation."""
    #     if self.explain_mode:
    #         msg = "Explain mode is not supported for this model."
    #         raise NotImplementedError(msg)

    #     return self.model(pixel_values=image)


if __name__ == "__main__":
    data_root = "/home/harimkan/workspace/repo/datasets/otx_v2_dataset/multiclass_classification/multiclass_CUB_medium"
    from otx.engine.utils.auto_configurator import AutoConfigurator
    dm = AutoConfigurator(data_root=data_root).get_datamodule()
    clip = CLIP("openai/clip-vit-base-patch32", label_info=dm.label_info)

    from otx.engine import Engine
    engine = Engine(model=clip, datamodule=dm)
    engine.train()
