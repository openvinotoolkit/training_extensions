# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import torch

from mmcv import ConfigDict
from mmdet.datasets import build_dataset
from otx.mpa.det.stage import DetectionStage
from otx.mpa.utils.config_utils import update_or_add_custom_hook
from otx.mpa.utils.logger import get_logger

logger = get_logger()


class IncrDetectionStage(DetectionStage):
    """Patch config to support incremental learning for object detection"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def configure_task(self, cfg, training, **kwargs):
        """Patch config to support incremental learning
        """
        logger.info(f'Incremental task config!!!!: training={training}')
        if 'task_adapt' in cfg:
            task_adapt_type = cfg['task_adapt'].get('type', None)
            task_adapt_op = cfg['task_adapt'].get('op', 'REPLACE')

            org_model_classes, model_classes, data_classes = \
                self.configure_classes(cfg, task_adapt_type, task_adapt_op)
            if data_classes != model_classes:
                self.configure_task_data_pipeline(cfg, model_classes, data_classes)
            # TODO[JAEGUK]: configure_anchor is not working
            if cfg['task_adapt'].get('use_mpa_anchor', False):
                self.configure_anchor(cfg)
            self.configure_task_cls_incr(cfg, task_adapt_type, org_model_classes, model_classes)

    def configure_classes(self, cfg, task_adapt_type, task_adapt_op):
        """Patch classes for model and dataset."""
        org_model_classes = self.get_model_classes(cfg)
        data_classes = self.get_data_classes(cfg)

        # Model classes
        if task_adapt_op == 'REPLACE':
            if len(data_classes) == 0:
                model_classes = org_model_classes.copy()
            else:
                model_classes = data_classes.copy()
        elif task_adapt_op == 'MERGE':
            model_classes = org_model_classes + [cls for cls in data_classes if cls not in org_model_classes]
        else:
            raise KeyError(f'{task_adapt_op} is not supported for task_adapt options!')

        if task_adapt_type == 'mpa':
            data_classes = model_classes
        cfg.task_adapt.final = model_classes
        cfg.model.task_adapt = ConfigDict(
            src_classes=org_model_classes,
            dst_classes=model_classes,
        )

        # Model architecture
        head_names = ('mask_head', 'bbox_head', 'segm_head')
        num_classes = len(model_classes)
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

        # Eval datasets
        if cfg.get('task', 'detection') == 'detection':
            eval_types = ['val', 'test']
            for eval_type in eval_types:
                if cfg.data[eval_type]['type'] == 'TaskAdaptEvalDataset':
                    cfg.data[eval_type]['model_classes'] = model_classes
                else:
                    # Wrap original dataset config
                    org_type = cfg.data[eval_type]['type']
                    cfg.data[eval_type]['type'] = 'TaskAdaptEvalDataset'
                    cfg.data[eval_type]['org_type'] = org_type
                    cfg.data[eval_type]['model_classes'] = model_classes

        return org_model_classes, model_classes, data_classes

    def configure_task_data_pipeline(self, cfg, model_classes, data_classes):
        # Trying to alter class indices of training data according to model class order
        tr_data_cfg = self.get_train_data_cfg(cfg)
        class_adapt_cfg = dict(type='AdaptClassLabels', src_classes=data_classes, dst_classes=model_classes)
        pipeline_cfg = tr_data_cfg.pipeline
        for i, op in enumerate(pipeline_cfg):
            if op['type'] == 'LoadAnnotations':  # insert just after this op
                op_next_ann = pipeline_cfg[i + 1] if i + 1 < len(pipeline_cfg) else {}
                if op_next_ann.get('type', '') == class_adapt_cfg['type']:
                    op_next_ann.update(class_adapt_cfg)
                else:
                    pipeline_cfg.insert(i + 1, class_adapt_cfg)
                break

    def configure_anchor(self, cfg, proposal_ratio=None):
        if cfg.model.type in ['SingleStageDetector', 'CustomSingleStageDetector']:
            anchor_cfg = cfg.model.bbox_head.anchor_generator
            if anchor_cfg.type == 'SSDAnchorGeneratorClustered':
                cfg.model.bbox_head.anchor_generator.pop('input_size', None)

    def configure_task_cls_incr(self, cfg, task_adapt_type, org_model_classes, model_classes):
        """Patch config for incremental learning"""
        if task_adapt_type == 'mpa':
            self.configure_bbox_head(cfg, model_classes)
            self.configure_task_adapt_hook(cfg, org_model_classes, model_classes)
            self.configure_ema(cfg)
            self.configure_val_interval(cfg)
        else:
            src_data_cfg = self.get_train_data_cfg(cfg)
            src_data_cfg.pop('old_new_indices', None)

    def configure_bbox_head(self, cfg, model_classes):
        """Patch bbox head in detector for class incremental learning.
        Most of patching are related with hyper-params in focal loss
        """
        if cfg.get('task', 'detection') == 'detection':
            bbox_head = cfg.model.bbox_head
        else:
            bbox_head = cfg.model.roi_head.bbox_head

        # TODO Remove this part
        # This is not related with patching bbox head
        # This might be useless when semisl using MPADetDataset
        tr_data_cfg = self.get_train_data_cfg(cfg)
        if tr_data_cfg.type != 'MPADetDataset':
            tr_data_cfg.img_ids_dict = self.get_img_ids_for_incr(cfg, org_model_classes, model_classes)
            tr_data_cfg.org_type = tr_data_cfg.type
            tr_data_cfg.type = 'DetIncrCocoDataset'

        alpha, gamma = 0.25, 2.0
        if bbox_head.type in ['SSDHead', 'CustomSSDHead']:
            gamma = 1 if cfg['task_adapt'].get('efficient_mode', False) else 2
            bbox_head.type = 'CustomSSDHead'
            bbox_head.loss_cls = ConfigDict(
                type='FocalLoss',
                loss_weight=1.0,
                gamma=gamma,
                reduction='none',
            )
        elif bbox_head.type in ['ATSSHead']:
            gamma = 3 if cfg['task_adapt'].get('efficient_mode', False) else 4.5
            bbox_head.loss_cls.gamma = gamma
        elif bbox_head.type in ['VFNetHead', 'CustomVFNetHead']:
            alpha = 0.75
            gamma = 1 if cfg['task_adapt'].get('efficient_mode', False) else 2
        # TODO Move this part
        # This is not related with patching bbox head
        elif bbox_head.type in ['YOLOXHead', 'CustomYOLOXHead']:
            if cfg.data.train.type == 'MultiImageMixDataset':
                cfg.data.train.pop('ann_file', None)
                cfg.data.train.pop('img_prefix', None)
                cfg.data.train['labels'] = cfg.data.train.pop('labels', None)
                self.add_yolox_hooks(cfg)

        if cfg.get('ignore', False):
            bbox_head.loss_cls = ConfigDict(
                    type='CrossSigmoidFocalLoss',
                    use_sigmoid=True,
                    num_classes=len(model_classes),
                    alpha=alpha,
                    gamma=gamma
            )

    @staticmethod
    def configure_task_adapt_hook(cfg, org_model_classes, model_classes):
        """Add TaskAdaptHook for sampler."""
        sampler_flag = True
        if len(set(org_model_classes) & set(model_classes)) == 0 or set(org_model_classes) == set(model_classes):
            sampler_flag = False
        update_or_add_custom_hook(
            cfg,
            ConfigDict(
                type='TaskAdaptHook',
                src_classes=org_model_classes,
                dst_classes=model_classes,
                model_type=cfg.model.type,
                sampler_flag=sampler_flag,
                efficient_mode=cfg['task_adapt'].get('efficient_mode', False)
            )
        )

    @staticmethod
    def configure_ema(cfg):
        """Patch ema settings."""
        adaptive_ema = cfg.get('adaptive_ema', {})
        if adaptive_ema:
            update_or_add_custom_hook(
                cfg,
                ConfigDict(
                    type='CustomModelEMAHook',
                    priority='ABOVE_NORMAL',
                    resume_from=cfg.get("resume_from"),
                    **adaptive_ema
                )
            )
        else:
            update_or_add_custom_hook(cfg, ConfigDict(type='EMAHook', priority='ABOVE_NORMAL', resume_from=cfg.get("resume_from"), momentum=0.1))

    @staticmethod
    def configure_val_interval(cfg):
        """Patch validation interval."""
        adaptive_validation_interval = cfg.get('adaptive_validation_interval', {})
        if adaptive_validation_interval:
            update_or_add_custom_hook(
                cfg,
                ConfigDict(type='AdaptiveTrainSchedulingHook', **adaptive_validation_interval)
            )

    @staticmethod
    def get_img_ids_for_incr(cfg, org_model_classes, model_classes):
        # get image ids of old classes & new class
        # to setup experimental dataset (COCO format)
        new_classes = np.setdiff1d(model_classes, org_model_classes).tolist()
        old_classes = np.intersect1d(org_model_classes, model_classes).tolist()

        src_data_cfg = self.get_train_data_cfg(cfg)

        ids_old, ids_new = [], []
        data_cfg = cfg.data.test.copy()
        data_cfg.test_mode = src_data_cfg.get('test_mode', False)
        data_cfg.ann_file = src_data_cfg.get('ann_file', None)
        data_cfg.img_prefix = src_data_cfg.get('img_prefix', None)
        old_data_cfg = data_cfg.copy()
        if 'classes' in old_data_cfg:
            old_data_cfg.classes = old_classes
        old_dataset = build_dataset(old_data_cfg)
        ids_old = old_dataset.dataset.img_ids
        if len(new_classes) > 0:
            data_cfg.classes = new_classes
            dataset = build_dataset(data_cfg)
            ids_new = dataset.dataset.img_ids
            ids_old = np.setdiff1d(ids_old, ids_new).tolist()

        sampled_ids = ids_old + ids_new
        outputs = dict(
            old_classes=old_classes,
            new_classes=new_classes,
            img_ids=sampled_ids,
            img_ids_old=ids_old,
            img_ids_new=ids_new,
        )
        return outputs

    @staticmethod
    def add_yolox_hooks(cfg):
        update_or_add_custom_hook(
            cfg,
            ConfigDict(
                type='YOLOXModeSwitchHook',
                num_last_epochs=15,
                priority=48))
        update_or_add_custom_hook(
            cfg,
            ConfigDict(
                type='SyncRandomSizeHook',
                ratio_range=(10, 20),
                img_scale=(640, 640),
                interval=1,
                priority=48,
                device='cuda' if torch.cuda.is_available() else 'cpu'))
        update_or_add_custom_hook(
            cfg,
            ConfigDict(
                type='SyncNormHook',
                num_last_epochs=15,
                interval=1,
                priority=48))
