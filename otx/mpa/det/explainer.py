# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmcv.utils import Config, ConfigDict
from mmdet.datasets import build_dataloader as mmdet_build_dataloader
from mmdet.datasets import build_dataset as mmdet_build_dataset
from mmdet.datasets import replace_ImageToTensor

from otx.algorithms.common.adapters.mmcv.hooks.recording_forward_hook import (
    ActivationMapHook,
    EigenCamHook,
)
from otx.algorithms.common.adapters.mmcv.utils import (
    build_data_parallel,
    build_dataloader,
    build_dataset,
)
from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.detection.adapters.mmdet.datasets import ImageTilingDataset
from otx.algorithms.detection.adapters.mmdet.hooks.det_saliency_map_hook import (
    DetSaliencyMapHook,
)
from otx.mpa.registry import STAGES

from .stage import DetectionStage

logger = get_logger()
EXPLAINER_HOOK_SELECTOR = {
    "classwisesaliencymap": DetSaliencyMapHook,
    "eigencam": EigenCamHook,
    "activationmap": ActivationMapHook,
}


@STAGES.register_module()
class DetectionExplainer(DetectionStage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = None

    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run explain stage for detection

        - Configuration
        - Environment setup
        - Run inference via MMDetection -> MMCV
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

        data_cfg = cfg.data.test.copy()

        # Input source
        if "input_source" in cfg:
            input_source = cfg.get("input_source")
            logger.info(f"Inferring on input source: data.{input_source}")
            src_data_cfg = cfg.data[input_source]
            data_cfg.test_mode = src_data_cfg.get("test_mode", False)
            data_cfg.ann_file = src_data_cfg.ann_file
            data_cfg.img_prefix = src_data_cfg.img_prefix
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
        samples_per_gpu = data_cfg.data.test_dataloader.get(
            "samples_per_gpu",
            data_cfg.data.get("samples_per_gpu", 1),
        )
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            data_cfg.data.test.pipeline = replace_ImageToTensor(data_cfg.data.test.pipeline)

        # Data loader
        self.dataset = build_dataset(data_cfg, "test", mmdet_build_dataset)
        test_dataloader = build_dataloader(
            self.dataset,
            data_cfg,
            "test",
            mmdet_build_dataloader,
            distributed=False,
        )

        # Target classes
        target_classes = self.dataset.CLASSES
        head_names = ("mask_head", "bbox_head", "segm_head")
        num_classes = len(target_classes)
        if "roi_head" in cfg.model:
            # For Faster-RCNNs
            for head_name in head_names:
                if head_name in cfg.model.roi_head:
                    if isinstance(cfg.model.roi_head[head_name], list):
                        for head in cfg.model.roi_head[head_name]:
                            head.num_classes = num_classes
                    else:
                        cfg.model.roi_head[head_name].num_classes = num_classes
        else:
            # For other architectures (including SSD)
            for head_name in head_names:
                if head_name in cfg.model:
                    cfg.model[head_name].num_classes = num_classes

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
        # TODO: Check Inference FP16 Support
        model = self.build_model(cfg, model_builder, fp16=False)
        model.CLASSES = target_classes
        model.eval()
        feature_model = self._get_feature_module(model)
        model = build_data_parallel(model, cfg, distributed=False)

        # InferenceProgressCallback (Time Monitor enable into Infer task)
        self.set_inference_progress_callback(model, cfg)

        # Class-wise Saliency map for Single-Stage Detector, otherwise use class-ignore saliency map.
        eval_predictions = []
        with self.explainer_hook(feature_model) as saliency_hook:
            for data in test_dataloader:
                with torch.no_grad():
                    result = model(return_loss=False, rescale=True, **data)
                eval_predictions.extend(result)
            saliency_maps = saliency_hook.records

        # Check and unwrap ImageTilingDataset object from TaskAdaptEvalDataset
        dataset = self.dataset
        while hasattr(dataset, "dataset") and not isinstance(dataset, ImageTilingDataset):
            dataset = dataset.dataset

        # In the tiling case, select the first images which is map of the entire image
        if isinstance(dataset, ImageTilingDataset):
            saliency_maps = [saliency_maps[i] for i in range(dataset.num_samples)]

        outputs = dict(detections=eval_predictions, saliency_maps=saliency_maps)
        return outputs
