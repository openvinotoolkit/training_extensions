# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import torch
from mmengine.config import ConfigDict
from otx.algo.instance_segmentation.heads.custom_rtmdet_ins_head import CustomRTMDetInsSepBNHead
from otx.algo.instance_segmentation.rtmdet_inst import RTMDetInst
from otx.core.types.export import OTXExportFormatType


class TestCustomRTMDetInsSepBNHead:
    def test_mask_pred(self, mocker) -> None:
        num_samples = 1
        num_classes = 1
        test_cfg = ConfigDict(
            nms_pre=100,
            score_thr=0.00,
            nms={"type": "nms", "iou_threshold": 0.5},
            max_per_img=100,
            mask_thr_binary=0.5,
            min_bbox_size=0,
        )
        s = 128
        img_metas = {
            "img_shape": (s, s, 3),
            "scale_factor": (1, 1),
            "ori_shape": (s, s, 3),
        }
        mask_head = CustomRTMDetInsSepBNHead(
            num_classes=num_classes,
            in_channels=1,
            num_prototypes=1,
            num_dyconvs=1,
            anchor_generator={
                "type": "MlvlPointGenerator",
                "offset": 0,
                "strides": (1,),
            },
            bbox_coder={"type": "DistancePointBBoxCoder"},
        )
        cls_scores = [torch.rand((num_samples, num_classes, 14, 14))]
        bbox_preds = [torch.rand((num_samples, 4, 14, 14))]
        kernel_preds = [torch.rand((1, 32, 14, 14))]
        mask_feat = torch.rand(num_samples, 1, 14, 14)

        mocker.patch.object(
            CustomRTMDetInsSepBNHead,
            "_mask_predict_by_feat_single",
            return_value=torch.rand(100, 14, 14),
        )

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

        mask_head._bbox_mask_post_process(
            results[0],
            mask_feat,
            cfg=None,
        )

    def test_predict_by_feat_ov(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdirname, torch.no_grad(), torch.device("cpu"):
            lit_module = RTMDetInst(num_classes=1, variant="tiny")
            exported_model_path = lit_module.export(
                output_dir=Path(tmpdirname),
                base_name="exported_model",
                export_format=OTXExportFormatType.OPENVINO,
            )
            Path.exists(exported_model_path)
