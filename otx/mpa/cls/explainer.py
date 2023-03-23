# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmcls.datasets import build_dataloader as mmcls_build_dataloader
from mmcls.datasets import build_dataset as mmcls_build_dataset

from otx.algorithms.common.adapters.mmcv.hooks.recording_forward_hook import (
    ActivationMapHook,
    EigenCamHook,
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
EXPLAINER_HOOK_SELECTOR = {
    "eigencam": EigenCamHook,
    "activationmap": ActivationMapHook,
    "classwisesaliencymap": ReciproCAMHook,
}


@STAGES.register_module()
class ClsExplainer(ClsStage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = None

    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run explain stage
        - Configuration
        - Environment setup
        - Run explain via hooks in recording_forward_hooks
        """
        self._init_logger()
        explainer = kwargs.get("explainer")
        self.explainer_hook = EXPLAINER_HOOK_SELECTOR.get(explainer.lower(), None)
        if self.explainer_hook is None:
            raise NotImplementedError(f"Explainer algorithm {explainer} not supported!")
        logger.info(f"Explainer algorithm: {explainer}")

        cfg = self.configure(model_cfg, model_ckpt, data_cfg, training=False, **kwargs)
        logger.info("explain!")

        model_builder = kwargs.get("model_builder", None)
        outputs = self.explain(cfg, model_builder)

        return dict(outputs=outputs)

    def explain(self, cfg, model_builder=None):
        # TODO: distributed inference

        # Data loader
        self.dataset = build_dataset(cfg, "test", mmcls_build_dataset)
        explain_data_loader = build_dataloader(
            self.dataset,
            cfg,
            "test",
            mmcls_build_dataloader,
            distributed=False,
            round_up=False,
        )
        self.configure_samples_per_gpu(cfg, "test", distributed=False)

        # build the model and load checkpoint
        model = self.build_model(cfg, model_builder, fp16=cfg.get("fp16", False))
        self.extract_prob = hasattr(model, "extract_prob")
        model.eval()
        feature_model = self._get_feature_module(model)
        model = build_data_parallel(model, cfg, distributed=False)

        # InferenceProgressCallback (Time Monitor enable into Infer task)
        self.set_inference_progress_callback(model, cfg)

        eval_predictions = []
        with self.explainer_hook(feature_model) as forward_explainer_hook:
            # do inference and record intermediate fmap
            for data in explain_data_loader:
                with torch.no_grad():
                    result = model(return_loss=False, **data)
                eval_predictions.extend(result)
            saliency_maps = forward_explainer_hook.records

        assert len(eval_predictions) == len(saliency_maps), (
            "Number of elements should be the same, however, number of outputs are "
            f"{len(eval_predictions)}, and {len(saliency_maps)}"
        )

        outputs = dict(
            eval_predictions=eval_predictions,
            saliency_maps=saliency_maps,
        )
        return outputs
