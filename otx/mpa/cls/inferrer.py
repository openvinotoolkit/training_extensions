# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os.path as osp
from contextlib import nullcontext

import mmcv
import numpy as np
import torch
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier
from mmcv.runner import load_checkpoint, wrap_fp16_model

from otx.mpa.cls.stage import ClsStage
from otx.mpa.modules.hooks.recording_forward_hooks import (
    FeatureVectorHook,
    ReciproCAMHook,
)
from otx.mpa.modules.utils.task_adapt import prob_extractor
from otx.mpa.registry import STAGES
from otx.mpa.utils.logger import get_logger

logger = get_logger()


@STAGES.register_module()
class ClsInferrer(ClsStage):
    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run inference stage

        - Configuration
        - Environment setup
        - Run inference via mmcls -> mmcv
        """
        self._init_logger()
        dump_features = kwargs.get("dump_features", False)
        dump_saliency_map = kwargs.get("dump_saliency_map", False)
        mode = kwargs.get("mode", "train")
        if mode not in self.mode:
            return {}

        cfg = self.configure(model_cfg, model_ckpt, data_cfg, training=False, **kwargs)

        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        outputs = self._infer(cfg, dump_features, dump_saliency_map)
        if cfg.get("task_adapt", False) and self.extract_prob:
            output_file_path = osp.join(cfg.work_dir, "pre_stage_res.npy")
            np.save(output_file_path, outputs, allow_pickle=True)
            return dict(pre_stage_res=output_file_path)
        else:
            # output_file_path = osp.join(cfg.work_dir, 'infer_result.npy')
            # np.save(output_file_path, outputs, allow_pickle=True)
            return dict(
                # output_file_path=output_file_path,
                outputs=outputs
            )

    def _infer(self, cfg, dump_features=False, dump_saliency_map=False):
        if cfg.get("task_adapt", False) and not hasattr(self, "eval"):
            dataset_cfg = cfg.data.train.copy()
            dataset_cfg.pipeline = cfg.data.test.pipeline
            self.dataset = build_dataset(dataset_cfg)
        else:
            self.dataset = build_dataset(cfg.data.test)

        # Data loader
        data_loader = build_dataloader(
            self.dataset,
            samples_per_gpu=cfg.data.samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False,
            round_up=False,
            persistent_workers=False,
        )

        # build the model and load checkpoint
        model = build_classifier(cfg.model)
        self.extract_prob = hasattr(model, "extract_prob")
        fp16_cfg = cfg.get("fp16", None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        if cfg.load_from is not None:
            logger.info("Load checkpoint from " + cfg.load_from)
            _ = load_checkpoint(model, cfg.load_from, map_location="cpu")

        model.eval()
        model = self._put_model_on_gpu(model, cfg)

        # InferenceProgressCallback (Time Monitor enable into Infer task)
        ClsStage.set_inference_progress_callback(model, cfg)
        eval_predictions = []
        feature_vectors = []
        saliency_maps = []

        if cfg.get("task_adapt", False) and not hasattr(self, "eval") and self.extract_prob:
            old_prob, feats = prob_extractor(model.module, data_loader)
            data_infos = self.dataset.data_infos
            # pre-stage for LwF
            for i, data_info in enumerate(data_infos):
                data_info["soft_label"] = {task: value[i] for task, value in old_prob.items()}
            outputs = data_infos
        else:
            with FeatureVectorHook(model.module) if dump_features else nullcontext() as feature_vector_hook:
                with ReciproCAMHook(model.module) if dump_saliency_map else nullcontext() as forward_explainer_hook:
                    for data in data_loader:
                        with torch.no_grad():
                            result = model(return_loss=False, **data)
                        eval_predictions.extend(result)
                    feature_vectors = feature_vector_hook.records if dump_features else [None] * len(self.dataset)
                    saliency_maps = forward_explainer_hook.records if dump_saliency_map else [None] * len(self.dataset)

        assert len(eval_predictions) == len(feature_vectors) == len(saliency_maps), (
            "Number of elements should be the same, however, number of outputs are "
            f"{len(eval_predictions)}, {len(feature_vectors)}, and {len(saliency_maps)}"
        )

        outputs = dict(eval_predictions=eval_predictions, feature_vectors=feature_vectors, saliency_maps=saliency_maps)
        return outputs
