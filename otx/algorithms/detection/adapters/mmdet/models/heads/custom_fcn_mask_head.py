import torch

from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.mask_heads.fcn_mask_head import FCNMaskHead


BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit


@HEADS.register_module()
class CustomFCNMaskHead(FCNMaskHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape (Tuple): original image height and width, shape (2,)
            scale_factor(ndarray | Tensor): If ``rescale is True``, box
                coordinates are divided by this scale factor to fit
                ``ori_shape``.
            rescale (bool): If True, the resulting masks will be rescaled to
                ``ori_shape``.

        Returns:
            list[list]: encoded masks. The c-th item in the outer list
                corresponds to the c-th class. Given the c-th outer list, the
                i-th item in that inner list is the mask for the i-th box with
                class label c.

        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid()
        else:
            # In AugTest, has been activated before
            mask_pred = det_bboxes.new_tensor(mask_pred)

        cls_segms = [[] for _ in range(self.num_classes)
                     ]  # BG is not included in num_classes
        labels = det_labels

        N = len(mask_pred)
        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.

        threshold = rcnn_test_cfg.mask_thr_binary

        if not self.class_agnostic:
            mask_pred = mask_pred[range(N), labels][:, None]

        for i in range(N):
            mask = mask_pred[i]
            if threshold >= 0:
                mask = (mask >= threshold).to(dtype=torch.bool)
            else:
                # for visualization and debugging
                mask = (mask * 255).to(dtype=torch.uint8)
            mask = mask.detach().cpu().numpy()
            cls_segms[labels[i]].append(mask[0])
        return cls_segms