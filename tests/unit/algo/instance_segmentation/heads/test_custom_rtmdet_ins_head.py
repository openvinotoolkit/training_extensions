# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from mmengine.config import ConfigDict
from otx.algo.instance_segmentation.heads.custom_rtmdet_ins_head import CustomRTMDetInsSepBNHead


class TestCustomRTMDetInsSepBNHead:
    def test_init(self) -> None:
        num_samples = 1
        num_classes = 1
        num_prototypes = 8
        test_cfg = ConfigDict(
            nms_pre=100, score_thr=0.05, nms=dict(type="nms", iou_threshold=0.5), max_per_img=100, mask_thr_binary=0.5
        )
        mask_head = CustomRTMDetInsSepBNHead(
            num_classes=num_classes,
            in_channels=1,
            anchor_generator={
                "type": "MlvlPointGenerator",
                "offset": 0,
                "strides": (1,),
            },
        )
        cls_scores = [torch.rand((num_samples, num_classes, 14, 14))]
        bbox_preds = [torch.rand((num_samples, 4, 14, 14))]
        kernel_preds = [torch.rand((num_samples, 100, 14, 14))]
        mask_feat = torch.rand(num_samples, num_prototypes, 14, 14)
        s = 128
        img_metas = {
            "img_shape": (s, s, 3),
            "scale_factor": (1, 1),
            "ori_shape": (s, s, 3),
        }

        mask_head.predict_by_feat(
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            kernel_preds=kernel_preds,
            mask_feat=mask_feat,
            batch_img_metas=[img_metas],
            cfg=test_cfg,
            rescale=True,
        )
