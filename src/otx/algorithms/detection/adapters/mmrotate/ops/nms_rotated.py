"""NMS Rotate Forward Rewriter."""
import torch
from mmdeploy.core.rewriters import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name="mmdeploy.mmcv.ops.nms_rotated.ONNXNMSRotatedOp.forward",
)
def nms_rotated__forward(ctx, self, boxes, scores, iou_threshold, score_threshold):
    """Rewrite Forward function of NMS Rotate.

    Note:
        score_threshold is hard-coded to 0.01 to prevent tracing error.

    Args:
        ctx (object): context object
        self (object): self object
        boxes (torch.Tensor): rboxes (x_ctr, y_ctr, w, h, angle)
        scores (torch.Tensor): scores
        iou_threshold (float): iou threshold
        score_threshold (float): score threshold

    Returns:
        indices: keep indices of boxes after nms
    """
    from mmcv.utils import ext_loader

    ext_module = ext_loader.load_ext("_ext", ["nms_rotated"])
    batch_size, num_class, _ = scores.shape

    assert batch_size == 1, "batch_size must be 1"
    score_threshold = 0.01
    indices = []
    batch_id = 0
    for cls_id in range(num_class):
        _boxes = boxes[batch_id, ...]
        # score_threshold=0 requires scores to be contiguous
        _scores = scores[batch_id, cls_id, ...].contiguous()
        valid_mask = _scores > score_threshold
        _boxes, _scores = _boxes[valid_mask], _scores[valid_mask]
        if _boxes.shape[0] == 0:
            continue
        valid_inds = torch.nonzero(valid_mask, as_tuple=False).squeeze(dim=1)
        _, order = _scores.sort(0, descending=True)
        dets_sorted = _boxes.index_select(0, order)
        box_inds = ext_module.nms_rotated(_boxes, _scores, order, dets_sorted, iou_threshold, 0)
        box_inds = valid_inds[box_inds]
        batch_inds = torch.zeros_like(box_inds) + batch_id
        cls_inds = torch.zeros_like(box_inds) + cls_id
        indices.append(torch.stack([batch_inds, cls_inds, box_inds], dim=-1))

    indices = torch.cat(indices)
    return indices
