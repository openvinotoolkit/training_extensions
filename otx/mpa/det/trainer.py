# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import glob
import os.path as osp
import time

from mmcv.utils import get_git_hash
from mmdet import __version__
from mmdet.apis import train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env
from torch import nn

from otx.mpa.det.incremental import IncrDetectionStage
from otx.mpa.modules.utils.task_adapt import extract_anchor_ratio
from otx.mpa.registry import STAGES
from otx.mpa.utils.logger import get_logger

from .stage import DetectionStage

# TODO[JAEGUK]: Remove import detection_tasks
# from detection_tasks.apis.detection.config_utils import cluster_anchors


logger = get_logger()


# FIXME DetectionTrainer does not inherit from stage
@STAGES.register_module()
class DetectionTrainer(IncrDetectionStage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run training stage for detection

        - Configuration
        - Environment setup
        - Run training via MMDetection -> MMCV
        """
        self._init_logger()
        mode = kwargs.get("mode", "train")
        if mode not in self.mode:
            return {}

        cfg = self.configure(model_cfg, model_ckpt, data_cfg, **kwargs)
        logger.info("train!")

        # # Work directory
        # mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

        # Environment
        logger.info(f"cfg.gpu_ids = {cfg.gpu_ids}, distributed = {self.distributed}")
        env_info_dict = collect_env()
        env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
        dash_line = "-" * 60 + "\n"
        logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)

        # Data
        datasets = [build_dataset(cfg.data.train)]
        cfg.data.val.samples_per_gpu = cfg.data.get("samples_per_gpu", 1)

        # FIXME: scale_factors is fixed at 1 even batch_size > 1 in simple_test_mask
        # Need to investigate, possibly due to OpenVINO
        if "roi_head" in model_cfg.model:
            if "mask_head" in model_cfg.model.roi_head:
                cfg.data.val.samples_per_gpu = 1

        if hasattr(cfg, "hparams"):
            if cfg.hparams.get("adaptive_anchor", False):
                num_ratios = cfg.hparams.get("num_anchor_ratios", 5)
                proposal_ratio = extract_anchor_ratio(datasets[0], num_ratios)
                self.configure_anchor(cfg, proposal_ratio)

        # Dataset for HPO
        hp_config = kwargs.get("hp_config", None)
        if hp_config is not None:
            import hpopt

            if isinstance(datasets[0], list):
                for idx, ds in enumerate(datasets[0]):
                    datasets[0][idx] = hpopt.createHpoDataset(ds, hp_config)
            else:
                datasets[0] = hpopt.createHpoDataset(datasets[0], hp_config)

        # Target classes
        if "task_adapt" in cfg:
            target_classes = cfg.task_adapt.get("final", [])
        else:
            target_classes = datasets[0].CLASSES

        # Metadata
        meta = dict()
        meta["env_info"] = env_info
        # meta['config'] = cfg.pretty_text
        meta["seed"] = cfg.seed
        meta["exp_name"] = cfg.work_dir
        if cfg.checkpoint_config is not None:
            cfg.checkpoint_config.meta = dict(mmdet_version=__version__ + get_git_hash()[:7], CLASSES=target_classes)
            if "proposal_ratio" in locals():
                cfg.checkpoint_config.meta.update({"anchor_ratio": proposal_ratio})

        # Save config
        # cfg.dump(osp.join(cfg.work_dir, 'config.py'))
        # logger.info(f'Config:\n{cfg.pretty_text}')

        model = build_detector(cfg.model)
        model.CLASSES = target_classes

        if self.distributed:
            self._modify_cfg_for_distributed(model, cfg)

        # Do clustering for SSD model
        # TODO[JAEGUK]: Temporary disable cluster_anchors for SSD model
        # if hasattr(cfg.model, 'bbox_head') and hasattr(cfg.model.bbox_head, 'anchor_generator'):
        #     if getattr(cfg.model.bbox_head.anchor_generator, 'reclustering_anchors', False):
        #         train_cfg = Stage.get_data_cfg(cfg, "train")
        #         train_dataset = train_cfg.get('otx_dataset', None)
        #         cfg, model = cluster_anchors(cfg, train_dataset, model)

        train_detector(
            model, datasets, cfg, distributed=self.distributed, validate=True, timestamp=timestamp, meta=meta
        )

        # Save outputs
        output_ckpt_path = osp.join(cfg.work_dir, "latest.pth")
        best_ckpt_path = glob.glob(osp.join(cfg.work_dir, "best_*.pth"))
        if len(best_ckpt_path) > 0:
            output_ckpt_path = best_ckpt_path[0]
        return dict(final_ckpt=output_ckpt_path)

    def _modify_cfg_for_distributed(self, model, cfg):
        nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if cfg.dist_params.get("linear_scale_lr", False):
            new_lr = len(cfg.gpu_ids) * cfg.optimizer.lr
            logger.info(
                f"enabled linear scaling rule to the learning rate. \
                changed LR from {cfg.optimizer.lr} to {new_lr}"
            )
            cfg.optimizer.lr = new_lr
