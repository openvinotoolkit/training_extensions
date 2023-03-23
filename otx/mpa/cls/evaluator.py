# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os.path as osp

from otx.algorithms.common.utils.logger import get_logger
from otx.mpa.registry import STAGES

from .inferrer import ClsInferrer

logger = get_logger()


@STAGES.register_module()
class ClsEvaluator(ClsInferrer):
    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run evaluation stage for classification

        - Run inference
        - Run evaluation via MMClassification -> MMCV
        """
        self._init_logger()
        mode = kwargs.get("mode", "train")
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

        # Evaluate inference results
        eval_cfg = cfg.get("evaluation", {}).copy()
        eval_cfg.pop("by_epoch", None)
        eval_result = self.dataset.evaluate(infer_results, **eval_cfg)

        logger.info(eval_result)
        return eval_result
