"""OTX MaskFormerFusionHead for Mask2Former Class for mmdetection detectors."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import torch.nn.functional as F
from mmdet.models.builder import HEADS
from mmdet.models.seg_heads import MaskFormerFusionHead


@HEADS.register_module()
class CustomMaskFormerFusionHead(MaskFormerFusionHead):
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

            if rescale:
                # return result in original resolution
                ori_height, ori_width = meta["ori_shape"][:2]
                mask_pred_result = F.interpolate(
                    mask_pred_result[:, None], size=(ori_height, ori_width), mode="bilinear", align_corners=False
                )[:, 0]
                # NOTE: detach to avoid gpu overflow
                mask_pred_result = mask_pred_result.detach().cpu()
                mask_cls_result = mask_cls_result.detach().cpu()

            result = dict()
            if panoptic_on:
                pan_results = self.panoptic_postprocess(mask_cls_result, mask_pred_result)
                result["pan_results"] = pan_results

            if instance_on:
                ins_results = self.instance_postprocess(mask_cls_result, mask_pred_result)
                result["ins_results"] = ins_results

            if semantic_on:
                sem_results = self.semantic_postprocess(mask_cls_result, mask_pred_result)
                result["sem_results"] = sem_results

            results.append(result)

        return results
