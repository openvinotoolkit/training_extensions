"""Adapt AVARoIHead in mmaction into OTX."""
import torch
from mmaction.core.bbox import bbox2result
from mmaction.models.heads import AVARoIHead as MMAVARoIHead
from mmdet.models import HEADS as MMDET_HEADS
from torch.onnx import is_in_onnx_export


@MMDET_HEADS.register_module(force=True)
# pylint: disable=abstract-method, unused-argument
class AVARoIHead(MMAVARoIHead):
    """AVARoIHead for OTX."""

    def simple_test(self, x, proposal_list, img_metas, proposals=None, rescale=False, **kwargs):
        """This is temporary soluition, since otx.mmdet is differnt with latest mmdet."""
        assert self.with_bbox, "Bbox head must be implemented."

        if isinstance(x, tuple):
            x_shape = x[0].shape
        else:
            x_shape = x.shape

        assert x_shape[0] == 1, "only accept 1 sample at test mode"
        assert x_shape[0] == len(img_metas) == len(proposal_list)

        with torch.no_grad():
            det_bboxes, det_labels = self.simple_test_bboxes(
                x, img_metas, proposal_list, self.test_cfg, rescale=rescale
            )
            if is_in_onnx_export():
                return det_bboxes, det_labels
            bbox_results = bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes, thr=self.test_cfg.action_thr)
            return [bbox_results]
