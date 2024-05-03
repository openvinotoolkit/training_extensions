# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch

from omegaconf import DictConfig

from otx.algo.instance_segmentation.rtmdet_inst import MMDetRTMDetInstTiny
from otx.algo.instance_segmentation.mmdet.models.dense_heads.rtmdet_ins_head import RTMDetInsSepBNHead
from otx.core.types.export import OTXExportFormatType

from otx.algo.detection.backbones.cspnext import CSPNeXt
from otx.algo.detection.heads.base_sampler import PseudoSampler
from otx.algo.detection.heads.distance_point_bbox_coder import DistancePointBBoxCoder
from otx.algo.detection.heads.dynamic_soft_label_assigner import DynamicSoftLabelAssigner
from otx.algo.detection.heads.point_generator import MlvlPointGenerator
from otx.algo.detection.losses.cross_entropy_loss import CrossEntropyLoss
from otx.algo.detection.losses.dice_loss import DiceLoss
from otx.algo.detection.losses.gfocal_loss import QualityFocalLoss
from otx.algo.detection.losses.iou_loss import GIoULoss
from otx.algo.detection.necks.cspnext_pafpn import CSPNeXtPAFPN
from otx.algo.instance_segmentation.mmdet.models.dense_heads.rtmdet_ins_head import RTMDetInsSepBNHead
from otx.algo.instance_segmentation.mmdet.models.detectors.rtmdet import RTMDet
from otx.core.config.data import TileConfig
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.metrics.mean_ap import MaskRLEMeanAPCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.instance_segmentation import MMDetInstanceSegCompatibleModel


class TestRTMDetInst:
    def test_mask_pred(self, mocker) -> None:
        num_samples = 1
        num_classes = 1

        train_cfg = {
            "assigner": DynamicSoftLabelAssigner(topk=13),
            "sampler": PseudoSampler(),
            "allowed_border": -1,
            "pos_weight": -1,
            "debug": False,
        }

        test_cfg = DictConfig(
            {
                "nms_pre": 100,
                "score_thr": 0.0,
                "nms": {"type": "nms", "iou_threshold": 1.0},
                "max_per_img": 100,
                "mask_thr_binary": 0.0,
                "min_bbox_size": -1,
            },
        )
        s = 128
        img_metas = {
            "img_shape": (s, s, 3),
            "scale_factor": (1, 1),
            "ori_shape": (s, s, 3),
        }
        mask_head = RTMDetInsSepBNHead(
            num_classes=num_classes,
            in_channels=96,
            stacked_convs=2,
            share_conv=True,
            pred_kernel_size=1,
            feat_channels=96,
            act_cfg={"type": "SiLU", "inplace": True},
            norm_cfg={"type": "BN", "requires_grad": True},
            anchor_generator=MlvlPointGenerator(
                offset=0,
                strides=[8, 16, 32],
            ),
            bbox_coder=DistancePointBBoxCoder(),
            loss_centerness=CrossEntropyLoss(use_sigmoid=True, loss_weight=1.0),
            loss_cls=QualityFocalLoss(
                use_sigmoid=True,
                beta=2.0,
                loss_weight=1.0,
            ),
            loss_bbox=GIoULoss(loss_weight=2.0),
            loss_mask=DiceLoss(
                loss_weight=2.0,
                eps=5.0e-06,
                reduction="mean",
            ),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        cls_scores = [torch.rand((num_samples, num_classes, 14, 14))]
        bbox_preds = [torch.rand((num_samples, 4, 14, 14))]
        kernel_preds = [torch.rand((1, 32, 14, 14))]
        mask_feat = torch.rand(num_samples, 1, 14, 14)

        # mocker.patch.object(
        #     RTMDetInsSepBNHead,
        #     "_mask_predict_by_feat_single",
        #     return_value=torch.rand(100, 14, 14),
        # )

        results = mask_head.predict_by_feat(
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            kernel_preds=kernel_preds,
            mask_feat=mask_feat,
            batch_img_metas=[img_metas],
            cfg=test_cfg,
            rescale=True,
        )

        mask_head._bbox_mask_post_process(
            results[0],
            mask_feat,
            cfg=test_cfg,
        )

    def test_predict_by_feat_ov(self, tmpdir) -> None:
        lit_module = MMDetRTMDetInstTiny(label_info=1)
        exported_model_path = lit_module.export(
            output_dir=Path(tmpdir),
            base_name="exported_model",
            export_format=OTXExportFormatType.OPENVINO,
        )
        Path.exists(exported_model_path)
