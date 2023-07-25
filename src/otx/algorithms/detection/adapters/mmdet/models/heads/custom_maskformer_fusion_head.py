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

    def simple_test(self, mask_cls_results, mask_pred_results, img_metas, rescale=False):
        """Test segment without test-time aumengtation.

        Only the output of last decoder layers was used.

        Args:
            mask_cls_results (Tensor): Mask classification logits,
                shape (batch_size, num_queries, cls_out_channels).
                Note `cls_out_channels` should includes background.
            mask_pred_results (Tensor): Mask logits, shape
                (batch_size, num_queries, h, w).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): If True, return boxes in
                original image space. Default False.

        Returns:
            list[dict[str, Tensor | tuple[Tensor]]]: Semantic segmentation \
                results and panoptic segmentation results for each \
                image.

            .. code-block:: none

                [
                    {
                        'pan_results': Tensor, # shape = [h, w]
                        'ins_results': tuple[Tensor],
                        # semantic segmentation results are not supported yet
                        'sem_results': Tensor
                    },
                    ...
                ]
        """
        panoptic_on = self.test_cfg.get("panoptic_on", True)
        semantic_on = self.test_cfg.get("semantic_on", False)
        instance_on = self.test_cfg.get("instance_on", False)
        assert not semantic_on, "segmantic segmentation " "results are not supported yet."

        results = []
        for mask_cls_result, mask_pred_result, meta in zip(mask_cls_results, mask_pred_results, img_metas):
            # remove padding
            img_height, img_width = meta["img_shape"][:2]
            mask_pred_result = mask_pred_result[:, :img_height, :img_width]

            result = dict()
            if panoptic_on:
                raise NotImplementedError

            if instance_on:
                ins_results = self.instance_postprocess(mask_cls_result, mask_pred_result, meta, rescale)
                result["ins_results"] = ins_results

            if semantic_on:
                raise NotImplementedError

            results.append(result)

        return results

    def instance_postprocess(self, mask_cls, mask_pred, meta, rescale=False):
        """Instance Segmentation postprocess.

        Args:
            mask_cls (_type_): _description_
            mask_pred (_type_): _description_
            meta (_type_): _description_
            rescale (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        max_per_image = self.test_cfg.get("max_per_image", 100)
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

        mask_pred_binary = (mask_pred > 0).float()
        mask_scores_per_image = (mask_pred.sigmoid() * mask_pred_binary).flatten(1).sum(1) / (
            mask_pred_binary.flatten(1).sum(1) + 1e-6
        )
        det_scores = scores_per_image * mask_scores_per_image
        mask_pred_binary = mask_pred_binary.bool()

        # filter by score
        keep = det_scores > self.test_cfg.score_threshold
        det_scores = det_scores[keep]
        mask_pred_binary = mask_pred_binary[keep]
        labels_per_image = labels_per_image[keep]

        # filter by mask area
        mask_area = mask_pred_binary.sum((1, 2))
        keep = mask_area > 0
        det_scores = det_scores[keep]
        mask_pred_binary = mask_pred_binary[keep]
        labels_per_image = labels_per_image[keep]

        if rescale:
            # NOTE: could switch to cpu if GPU OOM
            # mask_pred_binary = mask_pred_binary.detach().cpu()
            # det_scores = det_scores.detach().cpu()

            # return result in original resolution
            ori_height, ori_width = meta["ori_shape"][:2]
            mask_pred_binary = F.interpolate(
                (mask_pred_binary[:, None]).float(), size=(ori_height, ori_width), mode="bilinear", align_corners=False
            )[:, 0]
            mask_pred_binary = mask_pred_binary > 0

        bboxes = mask2bbox(mask_pred_binary)
        bboxes = torch.cat([bboxes, det_scores[:, None]], dim=-1)
        return labels_per_image, bboxes, mask_pred_binary
