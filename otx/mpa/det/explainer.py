# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import torch
from mmcv.runner import load_checkpoint

from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor, ImageTilingDataset
from mmdet.models import build_detector
from mmdet.utils.misc import prepare_mmdet_model_for_execution

from otx.mpa.det.stage import DetectionStage
from otx.mpa.modules.hooks.recording_forward_hooks import ActivationMapHook, EigenCamHook, DetSaliencyMapHook
from otx.mpa.registry import STAGES
from otx.mpa.utils.logger import get_logger
logger = get_logger()
EXPLAINER_HOOK_SELECTOR = {
    'classwisesaliencymap': DetSaliencyMapHook,
    'eigencam': EigenCamHook,
    'activationmap': ActivationMapHook,
}


@STAGES.register_module()
class DetectionExplainer(DetectionStage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run explain stage for detection

        - Configuration
        - Environment setup
        - Run inference via MMDetection -> MMCV
        """
        self._init_logger()
        explainer = kwargs.get('explainer')
        self.explainer_hook = EXPLAINER_HOOK_SELECTOR.get(explainer.lower(), None)
        if self.explainer_hook is None:
            raise NotImplementedError(f'Explainer algorithm {explainer} not supported!')
        logger.info(f'Explainer algorithm: {explainer}')
        cfg = self.configure(model_cfg, model_ckpt, data_cfg, training=False, **kwargs)
        outputs = self.explain(cfg)

        return dict(
            outputs=outputs
        )

    def explain(self, cfg):
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

        data_cfg = cfg.data.test.copy()
        # Input source
        if 'input_source' in cfg:
            input_source = cfg.get('input_source')
            logger.info(f'Inferring on input source: data.{input_source}')
            src_data_cfg = cfg.data[input_source]
            data_cfg.test_mode = src_data_cfg.get('test_mode', False)
            data_cfg.ann_file = src_data_cfg.ann_file
            data_cfg.img_prefix = src_data_cfg.img_prefix
            if 'classes' in src_data_cfg:
                data_cfg.classes = src_data_cfg.classes
        self.explain_dataset = build_dataset(data_cfg)
        explain_dataset = self.explain_dataset

        # Data loader
        explain_data_loader = build_dataloader(
            explain_dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)

        # Target classes
        target_classes = explain_dataset.CLASSES
        head_names = ('mask_head', 'bbox_head', 'segm_head')
        num_classes = len(target_classes)
        if 'roi_head' in cfg.model:
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
        if cfg.model.get('neck'):
            if isinstance(cfg.model.neck, list):
                for neck_cfg in cfg.model.neck:
                    if neck_cfg.get('rfp_backbone'):
                        if neck_cfg.rfp_backbone.get('pretrained'):
                            neck_cfg.rfp_backbone.pretrained = None
            elif cfg.model.neck.get('rfp_backbone'):
                if cfg.model.neck.rfp_backbone.get('pretrained'):
                    cfg.model.neck.rfp_backbone.pretrained = None

        model = build_detector(cfg.model)
        model.CLASSES = target_classes

        DetectionStage.set_inference_progress_callback(model, cfg)

        # Checkpoint
        if cfg.get('load_from', None):
            load_checkpoint(model, cfg.load_from, map_location='cpu')

        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        eval_model = prepare_mmdet_model_for_execution(model, cfg, self.distributed)

        # Class-wise Saliency map for Single-Stage Detector, otherwise use class-ignore saliency map.
        with self.explainer_hook(eval_model.module) as saliency_hook:
            for data in explain_data_loader:
                _ = eval_model(return_loss=False, rescale=True, **data)
            saliency_maps = saliency_hook.records

        # Check and unwrap ImageTilingDataset object from TaskAdaptEvalDataset
        while hasattr(explain_dataset, 'dataset') and not isinstance(explain_dataset, ImageTilingDataset):
            explain_dataset = explain_dataset.dataset

        # In the tiling case, select the first images which is map of the entire image
        if isinstance(explain_dataset, ImageTilingDataset):
            saliency_maps = [saliency_maps[i] for i in range(explain_dataset.num_samples)]

        outputs = dict(
            saliency_maps=saliency_maps
        )
        return outputs
