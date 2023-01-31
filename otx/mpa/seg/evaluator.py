# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os.path as osp

import torch

from otx.mpa.registry import STAGES
from otx.mpa.utils.logger import get_logger

from .inferrer import SegInferrer

logger = get_logger()


@STAGES.register_module()
class SegEvaluator(SegInferrer):
    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run evaluation stage for segmentation

        - Run inference
        - Run evaluation via MMSegmentation -> MMCV
        """
        self._init_logger()
        mode = kwargs.get("mode", "eval")
        if mode not in self.mode:
            logger.warning(f"Supported modes are {self.mode} but '{mode}' is given.")
            return {}

        cfg = self.configure(model_cfg, model_ckpt, data_cfg, training=False, **kwargs)
        logger.info("evaluate!")

        #  # Save config
        #  cfg.dump(osp.join(cfg.work_dir, "config.yaml"))
        #  logger.info(f"Config:\n{cfg.pretty_text}")

        # Inference
        model_builder = kwargs.get("model_builder", None)
        infer_results = super().infer(cfg, model_builder)
        segmentations = infer_results["segmentations"]

        # Change soft-prediction to hard-prediction
        hard_predictions = []
        for seg in segmentations:
            soft_prediction = torch.from_numpy(seg)
            hard_prediction = torch.argmax(soft_prediction, dim=0)
            if hard_prediction.device:
                hard_predictions.append(hard_prediction.numpy())
            else:
                hard_predictions.append(hard_prediction.cpu().detach().numpy())

        # Evaluate inference results
        eval_cfg = self.cfg.get("evaluation", {}).copy()
        for key in ["interval", "tmpdir", "start", "gpu_collect"]:
            eval_cfg.pop(key, None)
        eval_result = self.dataset.evaluate(hard_predictions, **eval_cfg)

        logger.info(eval_result)
        return dict(mAP=eval_result.get("bbox_mAP_50", 0.0))
