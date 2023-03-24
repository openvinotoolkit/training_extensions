# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from contextlib import nullcontext

import torch
from mmcv.runner import wrap_fp16_model
from mmcv.utils import Config, ConfigDict
from mmseg.datasets import build_dataloader as mmseg_build_dataloader
from mmseg.datasets import build_dataset as mmseg_build_dataset

from otx.algorithms.common.adapters.mmcv.hooks.recording_forward_hook import (
    FeatureVectorHook,
)
from otx.algorithms.common.adapters.mmcv.utils import (
    build_data_parallel,
    build_dataloader,
    build_dataset,
)
from otx.algorithms.common.utils.logger import get_logger
from otx.mpa.registry import STAGES
from otx.mpa.stage import Stage

from .stage import SegStage

logger = get_logger()


@STAGES.register_module()
class SegInferrer(SegStage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = None

    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run inference stage for segmentation

        - Configuration
        - Environment setup
        - Run inference via MMSegmentation -> MMCV
        """
        self._init_logger()
        mode = kwargs.get("mode", "train")
        if mode not in self.mode:
            logger.warning(f"Supported modes are {self.mode} but '{mode}' is given.")
            return {}

        cfg = self.configure(model_cfg, model_ckpt, data_cfg, training=False, **kwargs)
        logger.info("infer!")

        model_builder = kwargs.get("model_builder", None)
        dump_features = kwargs.get("dump_features", False)
        outputs = self.infer(
            cfg,
            model_builder=model_builder,
            dump_features=dump_features,
        )

        return dict(outputs=outputs)

    def infer(self, cfg, model_builder=None, dump_features=False):
        # TODO: distributed inference

        data_cfg = cfg.data.test.copy()

        # Input source
        input_source = cfg.get("input_source", "test")
        logger.info(f"Inferring on input source: data.{input_source}")
        if input_source == "train":
            src_data_cfg = Stage.get_data_cfg(cfg, "train")
        else:
            src_data_cfg = cfg.data[input_source]

        if "classes" in src_data_cfg:
            data_cfg.classes = src_data_cfg.classes

        data_cfg = Config(
            ConfigDict(
                data=ConfigDict(
                    samples_per_gpu=cfg.data.get("samples_per_gpu", 1),
                    workers_per_gpu=cfg.data.get("workers_per_gpu", 0),
                    test=data_cfg,
                    test_dataloader=cfg.data.get("test_dataloader", {}).copy(),
                ),
                gpu_ids=cfg.gpu_ids,
                seed=cfg.get("seed", None),
                model_task=cfg.model_task,
            )
        )
        self.configure_samples_per_gpu(data_cfg, "test", distributed=False)
        self.configure_compat_cfg(data_cfg)
        samples_per_gpu = data_cfg.data.test_dataloader.get("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            data_cfg.data.test.pipeline = replace_ImageToTensor(data_cfg.data.test.pipeline)

        # Data loader
        self.dataset = build_dataset(data_cfg, "test", mmseg_build_dataset)
        test_dataloader = build_dataloader(
            self.dataset,
            data_cfg,
            "test",
            mmseg_build_dataloader,
            distributed=False,
            # segmentor does not support various sized batch images
            samples_per_gpu=1,
        )

        # Target classes
        if "task_adapt" in cfg:
            target_classes = cfg.task_adapt.final
            if len(target_classes) < 1:
                raise KeyError(
                    f"target_classes={target_classes} is empty check the metadata from model ckpt or recipe "
                    "configuration"
                )
        else:
            target_classes = self.dataset.CLASSES

        # Model
        cfg.model.pretrained = None
        if cfg.model.get("neck"):
            if isinstance(cfg.model.neck, list):
                for neck_cfg in cfg.model.neck:
                    if neck_cfg.get("rfp_backbone"):
                        if neck_cfg.rfp_backbone.get("pretrained"):
                            neck_cfg.rfp_backbone.pretrained = None
            elif cfg.model.neck.get("rfp_backbone"):
                if cfg.model.neck.rfp_backbone.get("pretrained"):
                    cfg.model.neck.rfp_backbone.pretrained = None
        cfg.model.test_cfg.return_repr_vector = True
        model = self.build_model(cfg, model_builder, fp16=cfg.get("fp16", False))
        model.CLASSES = target_classes
        model.eval()
        feature_model = self._get_feature_module(model)
        model = build_data_parallel(model, cfg, distributed=False)

        # InferenceProgressCallback (Time Monitor enable into Infer task)
        self.set_inference_progress_callback(model, cfg)

        eval_predictions = []
        feature_vectors = []
        with FeatureVectorHook(feature_model) if dump_features else nullcontext() as fhook:
            for data in test_dataloader:
                with torch.no_grad():
                    result = model(return_loss=False, output_logits=True, **data)
                eval_predictions.append(result)
            feature_vectors = fhook.records if dump_features else [None] * len(self.dataset)

        assert len(eval_predictions) == len(feature_vectors), (
            "Number of elements should be the same, however, number of outputs are ",
            f"{len(eval_predictions)} and {len(feature_vectors)}",
        )

        outputs = dict(
            classes=target_classes,
            eval_predictions=eval_predictions,
            feature_vectors=feature_vectors,
        )
        return outputs


import copy  # noqa: E402
import warnings  # noqa: E402


def replace_ImageToTensor(pipelines):
    pipelines = copy.deepcopy(pipelines)
    for i, pipeline in enumerate(pipelines):
        if pipeline["type"] == "MultiScaleFlipAug":
            assert "transforms" in pipeline
            pipeline["transforms"] = replace_ImageToTensor(pipeline["transforms"])
        elif pipeline["type"] == "ImageToTensor":
            warnings.warn(
                '"ImageToTensor" pipeline is replaced by '
                '"DefaultFormatBundle" for batch inference. It is '
                "recommended to manually replace it in the test "
                "data pipeline in your config file.",
                UserWarning,
            )
            pipelines[i] = {"type": "DefaultFormatBundle"}
    return pipelines
