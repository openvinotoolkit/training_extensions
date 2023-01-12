# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import json
import os.path as osp

from otx.mpa.registry import STAGES
from otx.mpa.utils.logger import get_logger

from .inferrer import DetectionInferrer

logger = get_logger()


@STAGES.register_module()
class DetectionEvaluator(DetectionInferrer):
    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run evaluation stage for detection

        - Run inference
        - Run evaluation via MMDetection -> MMCV
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
        infer_results = self.infer(cfg, model_builder)
        detections = infer_results["detections"]

        # Evaluate inference results
        eval_cfg = self.cfg.get("evaluation", {}).copy()
        for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best"]:
            eval_cfg.pop(key, None)
        eval_result = self.dataset.evaluate(detections, **eval_cfg)

        output_file_path = osp.join(cfg.work_dir, "eval_result.json")
        with open(output_file_path, "w") as f:
            json.dump(eval_result, f, indent=4)

        logger.info(eval_result)
        return dict(mAP=eval_result.get("bbox_mAP_50", 0.0))
