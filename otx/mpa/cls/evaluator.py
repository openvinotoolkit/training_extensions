# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from os import path as osp

from otx.mpa.registry import STAGES
from .inferrer import ClsInferrer

from otx.mpa.utils.logger import get_logger

logger = get_logger()


@STAGES.register_module()
class ClsEvaluator(ClsInferrer):
    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run evaluation stage

        - Run inference
        - Run evaluation via MMDetection -> MMCV
        """
        self.eval = True
        self._init_logger()
        mode = kwargs.get('mode', 'train')
        if mode not in self.mode:
            logger.warning(f'mode for this stage {mode}')
            return {}

        cfg = self.configure(model_cfg, model_ckpt, data_cfg, training=False, **kwargs)

        # mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

        # Save config
        cfg.dump(osp.join(cfg.work_dir, 'config.yaml'))
        logger.info(f'Config:\n{cfg.pretty_text}')

        # Inference
        infer_results = super()._infer(cfg)

        eval_cfg = cfg.get('evaluation', {})
        eval_cfg.pop('by_epoch', False)
        results = self.dataset.evaluate(infer_results, **eval_cfg)
        logger.info(f'\n{results}')
        return results
