# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os.path as osp
import json
from otx.mpa.registry import STAGES
from otx.mpa.det.inferrer import DetectionInferrer
from otx.mpa.utils.logger import get_logger

logger = get_logger()


@STAGES.register_module()
class DetectionEvaluator(DetectionInferrer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run evaluation stage for detection

        - Run inference
        - Run evaluation via MMDetection -> MMCV
        """
        self._init_logger()
        mode = kwargs.get('mode', 'train')
        if mode not in self.mode:
            return {}

        cfg = self.configure(model_cfg, model_ckpt, data_cfg, training=False, **kwargs)
        cfg.dump(osp.join(cfg.work_dir, 'config.py'))
        logger.info(f'Config:\n{cfg.pretty_text}')
        logger.info('evaluate!')

        # mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

        # Inference
        infer_results = self.infer(cfg)
        detections = infer_results['detections']

        # Evaluate inference results
        eval_kwargs = self.cfg.get('evaluation', {}).copy()
        for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best']:
            eval_kwargs.pop(key, None)
        eval_result = self.dataset.evaluate(detections, **eval_kwargs)
        logger.info(eval_result)
        output_file_path = osp.join(cfg.work_dir, 'eval_result.json')
        with open(output_file_path, 'w') as f:
            json.dump(eval_result, f, indent=4)

        return dict(mAP=eval_result.get('bbox_mAP_50', 0.0))
