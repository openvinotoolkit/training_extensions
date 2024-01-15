# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for classification model entity used in OTX."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from pathlib import Path
from copy import copy

from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.classification import (
    MulticlassClsBatchDataEntity,
    MulticlassClsBatchPredEntity,
    MultilabelClsBatchDataEntity,
    MultilabelClsBatchPredEntity,
)
from otx.core.model.entity.base import OTXModel
from otx.core.utils.build import build_mm_model

if TYPE_CHECKING:
    from mmpretrain.models.utils import ClsDataPreprocessor
    from omegaconf import DictConfig
    from torch import device, nn


class OTXMulticlassClsModel(
    OTXModel[MulticlassClsBatchDataEntity, MulticlassClsBatchPredEntity],
):
    """Base class for the classification models used in OTX."""


def _create_mmpretrain_model(config: DictConfig, load_from: str) -> nn.Module:
    from mmpretrain.models.utils import ClsDataPreprocessor as _ClsDataPreprocessor
    from mmpretrain.registry import MODELS

    # NOTE: For the history of this monkey patching, please see
    # https://github.com/openvinotoolkit/training_extensions/issues/2743
    @MODELS.register_module(force=True)
    class ClsDataPreprocessor(_ClsDataPreprocessor):
        @property
        def device(self) -> device:
            try:
                buf = next(self.buffers())
            except StopIteration:
                return super().device
            else:
                return buf.device

    return build_mm_model(config, MODELS, load_from)


class MMPretrainMulticlassClsModel(OTXMulticlassClsModel):
    """Multi-class Classification model compatible for MMPretrain.

    It can consume MMPretrain model configuration translated into OTX configuration
    (please see otx.tools.translate_mmrecipe) and create the OTX classification model
    compatible for OTX pipelines.
    """

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.load_from = config.pop("load_from", None)
        super().__init__()

    def export(
            self,
            output_dir: Path | str,
            export_format: str = "ONNX",
            deploy_cfg: dict | None = None,
            precision: str = "fp32",
            test_pipeline: dict | None = None,
        ):
        """Export a PyTorch model for this class."""
        if deploy_cfg is None:
            raise NotImplementedError
        else:
            from otx.core.model.utils.mmdeploy import MMdeployExporter
            deploy_cfg = copy(deploy_cfg)

            if export_format == "ONNX":
                backend_cfg_backup = deploy_cfg["backend_config"]
                self._update_deploy_cfg_for_onnx(deploy_cfg)

            exporter = MMdeployExporter(self._create_model, output_dir, self.config, deploy_cfg, test_pipeline)
            exporter.cvt_torch2onnx()

            if export_format == "ONNX":
                pass
                # results["inference_parameters"] = {}
                # results["inference_parameters"]["mean_values"] = " ".join(
                #     map(str, backend_cfg_backup["mo_options"]["args"]["--mean_values"])
                # )
                # results["inference_parameters"]["scale_values"] = " ".join(
                #     map(str, backend_cfg_backup["mo_options"]["args"]["--scale_values"])
                # )

    @staticmethod
    def _update_deploy_cfg_for_onnx(deploy_cfg: dict):
        deploy_cfg["backend_config"] = {"type": "onnxruntime"}
        deploy_cfg["ir_config"]["dynamic_axes"]["data"] = {0: "batch"}

    def _create_model(self) -> nn.Module:
        return _create_mmpretrain_model(self.config, self.load_from)

    def _customize_inputs(self, entity: MulticlassClsBatchDataEntity) -> dict[str, Any]:
        from mmpretrain.structures import DataSample

        mmpretrain_inputs: dict[str, Any] = {}

        mmpretrain_inputs["inputs"] = entity.images  # B x C x H x W PyTorch tensor
        mmpretrain_inputs["data_samples"] = [
            DataSample(
                metainfo={
                    "img_id": img_info.img_idx,
                    "img_shape": img_info.img_shape,
                    "ori_shape": img_info.ori_shape,
                    "pad_shape": img_info.pad_shape,
                    "scale_factor": img_info.scale_factor,
                },
                gt_label=labels,
            )
            for img_info, labels in zip(
                entity.imgs_info,
                entity.labels,
            )
        ]
        preprocessor: ClsDataPreprocessor = self.model.data_preprocessor

        mmpretrain_inputs = preprocessor(data=mmpretrain_inputs, training=self.training)

        mmpretrain_inputs["mode"] = "loss" if self.training else "predict"
        return mmpretrain_inputs

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: MulticlassClsBatchDataEntity,
    ) -> MulticlassClsBatchPredEntity | OTXBatchLossEntity:
        from mmpretrain.structures import DataSample

        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            for k, v in outputs.items():
                losses[k] = v
            return losses

        scores = []
        labels = []

        for output in outputs:
            if not isinstance(output, DataSample):
                raise TypeError(output)

            scores.append(output.pred_score)
            labels.append(output.pred_label)

        return MulticlassClsBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            labels=labels,
        )


### NOTE, currently, although we've made the separate Multi-cls, Multi-label classes
### It'll be integrated after H-label classification integration with more advanced design.


class OTXMultilabelClsModel(
    OTXModel[MultilabelClsBatchDataEntity, MultilabelClsBatchPredEntity],
):
    """Multi-label classification models used in OTX."""


class MMPretrainMultilabelClsModel(OTXMultilabelClsModel):
    """Multi-label Classification model compatible for MMPretrain.

    It can consume MMPretrain model configuration translated into OTX configuration
    (please see otx.tools.translate_mmrecipe) and create the OTX classification model
    compatible for OTX pipelines.
    """

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.load_from = config.pop("load_from", None)
        super().__init__()

    def _create_model(self) -> nn.Module:
        return _create_mmpretrain_model(self.config, self.load_from)

    def _customize_inputs(self, entity: MultilabelClsBatchDataEntity) -> dict[str, Any]:
        from mmpretrain.structures import DataSample

        mmpretrain_inputs: dict[str, Any] = {}

        mmpretrain_inputs["inputs"] = entity.images  # B x C x H x W PyTorch tensor
        mmpretrain_inputs["data_samples"] = [
            DataSample(
                metainfo={
                    "img_id": img_info.img_idx,
                    "img_shape": img_info.img_shape,
                    "ori_shape": img_info.ori_shape,
                    "pad_shape": img_info.pad_shape,
                    "scale_factor": img_info.scale_factor,
                },
                gt_score=labels,
            )
            for img_info, labels in zip(
                entity.imgs_info,
                entity.labels,
            )
        ]
        preprocessor: ClsDataPreprocessor = self.model.data_preprocessor

        mmpretrain_inputs = preprocessor(data=mmpretrain_inputs, training=self.training)

        mmpretrain_inputs["mode"] = "loss" if self.training else "predict"
        return mmpretrain_inputs

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: MultilabelClsBatchDataEntity,
    ) -> MultilabelClsBatchPredEntity | OTXBatchLossEntity:
        from mmpretrain.structures import DataSample

        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            for k, v in outputs.items():
                losses[k] = v
            return losses

        scores = []
        labels = []

        for output in outputs:
            if not isinstance(output, DataSample):
                raise TypeError(output)

            scores.append(output.pred_score)
            labels.append(output.pred_label)

        return MultilabelClsBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            labels=labels,
        )


class OVClassificationCompatibleModel(OTXMulticlassClsModel):
    """Classification model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the OTX classification model compatible for OTX testing pipeline.
    """

    def __init__(self, config: DictConfig) -> None:
        self.model_name = config.pop("model_name")
        self.config = config
        super().__init__()

    def _create_model(self) -> nn.Module:
        from openvino.model_api.models import ClassificationModel

        return ClassificationModel.create_model(self.model_name, model_type="Classification")

    def _customize_inputs(self, entity: MulticlassClsBatchDataEntity) -> dict[str, Any]:
        if entity.batch_size > 1:
            msg = "Only sync inference with batch = 1 is supported for now"
            raise RuntimeError(msg)
        # restore original numpy image
        img = np.transpose(entity.images[-1].numpy(), (1, 2, 0))
        return {"inputs": img}

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: MulticlassClsBatchDataEntity,
    ) -> MulticlassClsBatchPredEntity:
        # add label index
        labels = [torch.tensor(outputs.top_labels[0][0], dtype=torch.long)]
        # add probability
        scores = [torch.tensor(outputs.top_labels[0][2])]

        return MulticlassClsBatchPredEntity(
            batch_size=1,
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            labels=labels,
        )
