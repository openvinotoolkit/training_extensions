# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os.path as osp
from contextlib import nullcontext

import numpy as np
import torch
from mmcls.datasets import build_dataloader as mmcls_build_dataloader
from mmcls.datasets import build_dataset as mmcls_build_dataset
from mmcv import Config, ConfigDict

from otx.algorithms import TRANSFORMER_BACKBONES
from otx.algorithms.common.adapters.mmcv.hooks.recording_forward_hook import (
    FeatureVectorHook,
    ReciproCAMHook,
)
from otx.algorithms.common.adapters.mmcv.utils import (
    build_data_parallel,
    build_dataloader,
    build_dataset,
)
from otx.algorithms.common.utils.logger import get_logger
from otx.mpa.registry import STAGES

from .stage import ClsStage

logger = get_logger()


@STAGES.register_module()
class ClsInferrer(ClsStage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = None

    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run inference stage for classification

        - Configuration
        - Environment setup
        - Run inference via MMClassification -> MMCV
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
        dump_saliency_map = kwargs.get("dump_saliency_map", False)
        # TODO: It looks like we need to modify that code in an appropriate way.
        if model_cfg.model.head.get("type", None) == "VisionTransformerClsHead":
            dump_features = False
            dump_saliency_map = False
        eval = kwargs.get("eval", False)
        outputs = self.infer(
            cfg,
            model_builder=model_builder,
            eval=eval,
            dump_features=dump_features,
            dump_saliency_map=dump_saliency_map,
        )

        return dict(outputs=outputs)

    def infer(self, cfg, model_builder=None, eval=False, dump_features=False, dump_saliency_map=False):
        # TODO: distributed inference

        if cfg.get("task_adapt", False) and not eval:
            data_cfg = cfg.data.train.copy()
            data_cfg.pipeline = cfg.data.test.pipeline
        else:
            data_cfg = cfg.data.test.copy()

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

        # Data loader
        self.dataset = build_dataset(data_cfg, "test", mmcls_build_dataset)
        test_dataloader = build_dataloader(
            self.dataset,
            data_cfg,
            "test",
            mmcls_build_dataloader,
            distributed=False,
            round_up=False,
        )

        # Model
        model = self.build_model(cfg, model_builder, fp16=cfg.get("fp16", False))
        model.eval()
        feature_model = self._get_feature_module(model)
        model = build_data_parallel(model, cfg, distributed=False)

        # InferenceProgressCallback (Time Monitor enable into Infer task)
        self.set_inference_progress_callback(model, cfg)

        eval_predictions = []
        feature_vectors = []
        saliency_maps = []
        with FeatureVectorHook(feature_model) if dump_features else nullcontext() as feature_vector_hook:
            with ReciproCAMHook(feature_model) if dump_saliency_map else nullcontext() as forward_explainer_hook:
                for data in test_dataloader:
                    with torch.no_grad():
                        result = model(return_loss=False, **data)
                    eval_predictions.extend(result)
                feature_vectors = feature_vector_hook.records if dump_features else [None] * len(self.dataset)
                saliency_maps = forward_explainer_hook.records if dump_saliency_map else [None] * len(self.dataset)
                if len(eval_predictions) == 0:
                    eval_predictions = [None] * len(self.dataset)

        assert len(eval_predictions) == len(feature_vectors) == len(saliency_maps), (
            "Number of elements should be the same, however, number of outputs are "
            f"{len(eval_predictions)}, {len(feature_vectors)}, and {len(saliency_maps)}"
        )

        outputs = dict(
            eval_predictions=eval_predictions,
            feature_vectors=feature_vectors,
            saliency_maps=saliency_maps,
        )
        return outputs
