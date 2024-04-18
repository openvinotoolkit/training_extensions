"""Base configurer for mmdet config."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

from mmcv.ops.nms import NMSop
from mmcv.ops.roi_align import RoIAlign
from mmcv.utils import ConfigDict

from otx.algorithms.common.adapters.mmcv.clsincr_mixin import IncrConfigurerMixin
from otx.algorithms.common.adapters.mmcv.configurer import BaseConfigurer
from otx.algorithms.common.adapters.mmcv.semisl_mixin import SemiSLConfigurerMixin
from otx.algorithms.common.adapters.mmcv.utils.config_utils import (
    InputSizeManager,
)
from otx.algorithms.detection.adapters.mmdet.utils import (
    cluster_anchors,
    monkey_patched_nms,
    monkey_patched_roi_align,
    patch_tiling,
    should_cluster_anchors,
)
from otx.utils.logger import get_logger

logger = get_logger()


# pylint: disable=too-many-public-methods
class DetectionConfigurer(BaseConfigurer):
    """Patch config to support otx train."""

    def override_from_hyperparams(self, config, hyperparams, **kwargs):
        """Override config using hyperparameters from OTX cli."""
        dataset = kwargs.get("train_dataset", None)
        super().override_from_hyperparams(config, hyperparams)
        patch_tiling(config, hyperparams, dataset)

    def configure_model(self, cfg, data_classes, model_classes, ir_options, **kwargs):
        """Configuration for model config."""
        super().configure_model(cfg, data_classes, model_classes, ir_options, **kwargs)
        self.configure_regularization(cfg)
        self.configure_max_num_detections(cfg, kwargs.get("max_num_detections", 0))
        self.configure_nms_iou_threshold(cfg, kwargs.get("nms_iou_threshold", 0.5))

    def configure_max_num_detections(self, cfg, max_num_detections):
        """Patch config for maximum number of detections."""
        if max_num_detections > 0:
            logger.info(f"Model max_num_detections: {max_num_detections}")
            test_cfg = cfg.model.test_cfg
            test_cfg.max_per_img = max_num_detections
            test_cfg.nms_pre = max_num_detections * 10
            # Special cases for 2-stage detectors (e.g. MaskRCNN)
            if hasattr(test_cfg, "rpn"):
                test_cfg.rpn.nms_pre = max_num_detections * 20
                test_cfg.rpn.max_per_img = max_num_detections * 10
            if hasattr(test_cfg, "rcnn"):
                test_cfg.rcnn.max_per_img = max_num_detections
            train_cfg = cfg.model.train_cfg
            if hasattr(train_cfg, "rpn_proposal"):
                train_cfg.rpn_proposal.nms_pre = max_num_detections * 20
                train_cfg.rpn_proposal.max_per_img = max_num_detections * 10

    def configure_nms_iou_threshold(self, cfg, nms_iou_threshold):
        """Configure nms iou threshold to user specified value if the object detector uses nms."""
        if "test_cfg" in cfg.model and "nms" in cfg.model.test_cfg:
            logger.info(
                "IoU NMS Threshold will be updated from "
                f"{cfg.model.test_cfg.nms.iou_threshold} --> {nms_iou_threshold}"
            )
            cfg.model.test_cfg.nms.iou_threshold = nms_iou_threshold
        elif "test_cfg" in cfg.model and "rcnn" in cfg.model.test_cfg and "nms" in cfg.model.test_cfg.rcnn:
            logger.info(
                "IoU NMS Threshold will be updated from "
                f"{cfg.model.test_cfg.rcnn.nms.iou_threshold} --> {nms_iou_threshold}"
            )
            cfg.model.test_cfg.rcnn.nms.iou_threshold = nms_iou_threshold
        else:
            logger.warning("Detector do not have nms postprocessing, user specified nms threshold will be omitted")
        if "tile_cfg" in cfg:
            cfg.tile_cfg.iou_threshold = nms_iou_threshold
            logger.info(
                "IoU NMS Threshold for tiling will be updated from "
                f"{cfg.tile_cfg.iou_threshold} --> {nms_iou_threshold}"
            )

    def configure_regularization(self, cfg):  # noqa: C901
        """Patch regularization parameters."""
        if self.training:
            if cfg.model.get("l2sp_weight", 0.0) > 0.0:
                logger.info("regularization config!!!!")

                # Checkpoint
                l2sp_ckpt = cfg.model.get("l2sp_ckpt", None)
                if l2sp_ckpt is None:
                    if "pretrained" in cfg.model:
                        l2sp_ckpt = cfg.model.pretrained
                    if cfg.load_from:
                        l2sp_ckpt = cfg.load_from
                cfg.model.l2sp_ckpt = l2sp_ckpt

                # Disable weight decay
                if "weight_decay" in cfg.optimizer:
                    cfg.optimizer.weight_decay = 0.0

    def configure_task(self, cfg, **kwargs):
        """Patch config to support training algorithm."""

        assert "train_dataset" in kwargs
        train_dataset = kwargs["train_dataset"]

        super().configure_task(cfg, **kwargs)
        if "task_adapt" in cfg:
            if self.data_classes != self.model_classes:
                self.configure_task_data_pipeline(cfg)
            if cfg["task_adapt"].get("use_adaptive_anchor", False):
                self.configure_anchor(cfg, train_dataset)
            if self.task_adapt_type == "default_task_adapt":
                self.configure_bbox_head(cfg)

    def configure_device(self, cfg):
        """Setting device for training and inference."""
        super().configure_device(cfg)

        patch_det_ops = cfg.device in ["xpu", "hpu"] or (
            cfg.model.backbone.get("type", "") and cfg.model.backbone.get("depth", 0) == 50
        )

        if patch_det_ops:
            NMSop.forward = monkey_patched_nms
            RoIAlign.forward = monkey_patched_roi_align

    def configure_classes(self, cfg):
        """Patch classes for model and dataset."""
        super().configure_classes(cfg)
        self._configure_eval_dataset(cfg)

    def _configure_head(self, cfg):
        """Patch number of classes of head."""
        head_names = ("mask_head", "bbox_head", "segm_head")
        num_classes = len(self.model_classes)
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

    def _configure_eval_dataset(self, cfg):
        if cfg.get("task", "detection") == "detection":
            eval_types = ["val", "test"]
            for eval_type in eval_types:
                if cfg.data[eval_type]["type"] == "TaskAdaptEvalDataset":
                    cfg.data[eval_type]["model_classes"] = self.model_classes
                else:
                    # Wrap original dataset config
                    org_type = cfg.data[eval_type]["type"]
                    cfg.data[eval_type]["type"] = "TaskAdaptEvalDataset"
                    cfg.data[eval_type]["org_type"] = org_type
                    cfg.data[eval_type]["model_classes"] = self.model_classes

    def configure_task_data_pipeline(self, cfg):
        """Trying to alter class indices of training data according to model class order."""
        tr_data_cfg = self.get_subset_data_cfg(cfg, "train")
        class_adapt_cfg = dict(type="AdaptClassLabels", src_classes=self.data_classes, dst_classes=self.model_classes)
        pipeline_cfg = tr_data_cfg.pipeline
        for i, operation in enumerate(pipeline_cfg):
            if operation["type"] in [
                "LoadAnnotationFromOTXDataset",
                "LoadResizeDataFromOTXDataset",
            ]:  # insert just after this operation
                op_next_ann = pipeline_cfg[i + 1] if i + 1 < len(pipeline_cfg) else {}
                if op_next_ann.get("type", "") == class_adapt_cfg["type"]:
                    op_next_ann.update(class_adapt_cfg)
                else:
                    pipeline_cfg.insert(i + 1, class_adapt_cfg)
                break

    def configure_anchor(self, cfg, train_dataset):
        """Patch anchor settings for single stage detector."""
        if cfg.model.type in ["SingleStageDetector", "CustomSingleStageDetector"]:
            anchor_cfg = cfg.model.bbox_head.anchor_generator
            if anchor_cfg.type == "SSDAnchorGeneratorClustered":
                cfg.model.bbox_head.anchor_generator.pop("input_size", None)
        if should_cluster_anchors(cfg) and train_dataset is not None:
            cluster_anchors(cfg, train_dataset)

    def configure_bbox_head(self, cfg):
        """Patch classification loss if there are ignore labels."""
        if cfg.get("task", "detection") == "detection":
            bbox_head = cfg.model.bbox_head
        else:
            bbox_head = cfg.model.roi_head.bbox_head

        if cfg.get("ignore", False):
            bbox_head.loss_cls = ConfigDict(
                type="CrossSigmoidFocalLoss",
                use_sigmoid=True,
                num_classes=len(self.model_classes),
                alpha=bbox_head.loss_cls.get("alpha", 0.25),
                gamma=bbox_head.loss_cls.get("gamma", 2.0),
            )

    @staticmethod
    def configure_input_size(
        cfg, input_size=Optional[Tuple[int, int]], model_ckpt_path: Optional[str] = None, training=True
    ):
        """Change input size if necessary."""
        if input_size is None:  # InputSizePreset.DEFAULT
            return

        # YOLOX tiny has a different input size in train and val data pipeline
        base_input_size = None
        model_cfg = cfg.get("model")
        if model_cfg is not None:
            if cfg.model.type == "CustomYOLOX" and cfg.model.backbone.widen_factor == 0.375:  # YOLOX tiny case
                base_input_size = {
                    "train": (640, 640),
                    "val": (416, 416),
                    "test": (416, 416),
                    "unlabeled": (992, 736),
                }
        manager = InputSizeManager(cfg, base_input_size)

        if input_size == (0, 0):  # InputSizePreset.AUTO
            if training:
                input_size = BaseConfigurer.adapt_input_size_to_dataset(cfg, manager, use_annotations=True)
            else:
                input_size = manager.get_trained_input_size(model_ckpt_path)
            if input_size is None:
                return

        manager.set_input_size(input_size)
        logger.info("Input size is changed to {}".format(input_size))


class IncrDetectionConfigurer(IncrConfigurerMixin, DetectionConfigurer):
    """Patch config to support incremental learning for object detection."""

    def configure_task(self, cfg, **kwargs):
        """Patch config to support incremental learning."""
        super(IncrConfigurerMixin, self).configure_task(cfg, **kwargs)
        if "task_adapt" in cfg and self.task_adapt_type == "default_task_adapt":
            self.configure_task_adapt_hook(cfg)


class SemiSLDetectionConfigurer(SemiSLConfigurerMixin, DetectionConfigurer):
    """Patch config to support semi supervised learning for object detection."""
