# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os.path as osp
import torch

import mmcv
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier

from otx.mpa.registry import STAGES
from otx.mpa.cls.stage import ClsStage
from otx.mpa.modules.hooks.recording_forward_hooks import ActivationMapHook, EigenCamHook, ReciproCAMHook
from otx.mpa.utils.logger import get_logger
logger = get_logger()
EXPLAINER_HOOK_SELECTOR = {
    'eigencam': EigenCamHook,
    'activationmap': ActivationMapHook,
    'classwisesaliencymap': ReciproCAMHook,
}


@STAGES.register_module()
class ClsExplainer(ClsStage):
    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run explain stage
        - Configuration
        - Environment setup
        - Run explain via hooks in recording_forward_hooks
        """
        self._init_logger()
        explainer = kwargs.get('explainer')
        self.explainer_hook = EXPLAINER_HOOK_SELECTOR.get(explainer.lower(), None)
        if self.explainer_hook is None:
            raise NotImplementedError(f'Explainer algorithm {explainer} not supported!')
        logger.info(
            f'Explainer algorithm: {explainer}'
        )
        cfg = self.configure(model_cfg, model_ckpt, data_cfg, training=False, **kwargs)

        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        outputs = self._explain(cfg)
        return dict(
            outputs=outputs
        )

    def _explain(self, cfg):
        self.explain_dataset = build_dataset(cfg.data.test)

        # Data loader
        explain_data_loader = build_dataloader(
            self.explain_dataset,
            samples_per_gpu=cfg.data.samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False,
            round_up=False,
            persistent_workers=False)

        # build the model and load checkpoint
        model = build_classifier(cfg.model)
        self.extract_prob = hasattr(model, 'extract_prob')
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        if cfg.load_from is not None:
            logger.info('Load checkpoint from ' + cfg.load_from)
            _ = load_checkpoint(model, cfg.load_from, map_location='cpu')

        model.eval()
        model = self._put_model_on_gpu(model, cfg)

        # InferenceProgressCallback (Time Monitor enable into Infer task)
        ClsStage.set_inference_progress_callback(model, cfg)
        with self.explainer_hook(model.module) as forward_explainer_hook:
            # do inference and record intermediate fmap
            for data in explain_data_loader:
                with torch.no_grad():
                    _ = model(return_loss=False, **data)
            saliency_maps = forward_explainer_hook.records

        outputs = dict(
            saliency_maps=saliency_maps
        )
        return outputs
