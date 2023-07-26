"""OTX MaskFormerFusionHead for Mask2Former Class for mmdetection detectors."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn.functional as F
from mmdet.core import mask2bbox
from mmdet.models.builder import HEADS
from mmdet.models.seg_heads import MaskFormerFusionHead


@HEADS.register_module()
class CustomMaskFormerFusionHead(MaskFormerFusionHead):
    """MaskFormerFusionHead for Mask2Former Class for mmdetection detectors."""

    def instance_postprocess(self, mask_cls, mask_pred):
        max_per_image = self.test_cfg.get('max_per_image', 100)
        num_queries = mask_cls.shape[0]
        # shape (num_queries, num_class)
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        # shape (num_queries * num_class, )
        labels = (
            torch.arange(self.num_classes, device=mask_cls.device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
        )
        scores_per_image, top_indices = scores.flatten(0, 1).topk(max_per_image, sorted=False)
        labels_per_image = labels[top_indices]

        query_indices = top_indices // self.num_classes
        mask_pred = mask_pred[query_indices]

        # extract things
        is_thing = labels_per_image < self.num_things_classes
        scores_per_image = scores_per_image[is_thing]
        labels_per_image = labels_per_image[is_thing]
        mask_pred = mask_pred[is_thing]
        mask_pred_binary = (mask_pred > 0).bool()

        # NOTE: Rewrite this part to avoid GPU OOM. ITS SLOWER BUT WORKS.
        bboxes = mask2bbox(mask_pred_binary)
        N = bboxes.shape[0]
        mask_scores_per_image = torch.zeros(N, device=mask_pred.device)
        for i in range(N):
            x1, y1, x2, y2 = list(map(int, bboxes[i]))
            score_mask = mask_pred[i, y1:y2, x1:x2]
            bool_mask = (score_mask > 0).float()
            if bool_mask.sum() > 0:
                mask_scores_per_image[i] = (score_mask.sigmoid() * bool_mask).sum() / (bool_mask.sum() + 1e-6)
            else:
                mask_scores_per_image[i] = 0.0
        det_scores = scores_per_image * mask_scores_per_image

        # filter by score
        keep = det_scores > self.test_cfg.score_threshold
        det_scores = det_scores[keep]
        mask_pred_binary = mask_pred_binary[keep]
        labels_per_image = labels_per_image[keep]
        bboxes = bboxes[keep]

        bboxes = torch.cat([bboxes, det_scores[:, None]], dim=-1)
        return labels_per_image, bboxes, mask_pred_binary
