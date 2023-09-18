"""RTMDet-Ins head."""
# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, bias_init_with_prob, constant_init, is_norm, normal_init
from mmcv.ops import RoIAlign, batched_nms
from mmcv.runner import BaseModule
from mmdet.core import bbox2roi
from mmdet.core.bbox import SamplingResult, distance2bbox
from mmdet.core.utils import filter_scores_and_topk, multi_apply, reduce_mean, select_single_mlvl
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.utils import sigmoid_geometric_mean
from mmdet.models.utils.transformer import inverse_sigmoid
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor, nn

from otx.algorithms.common.adapters.mmdeploy.utils import is_mmdeploy_enabled

from .rtmdet_head import RTMDetHead


def scale_boxes(boxes: Tensor, scale_factor: Tuple[float, float]) -> Tensor:
    """Scale boxes with type of tensor or box type.

    Args:
        boxes (Tensor or :obj:`BaseBoxes`): boxes need to be scaled. Its type
            can be a tensor or a box type.
        scale_factor (Tuple[float, float]): factors for scaling boxes.
            The length should be 2.

    Returns:
        Union[Tensor, :obj:`BaseBoxes`]: Scaled boxes.
    """
    # Tensor boxes will be treated as horizontal boxes
    repeat_num = int(boxes.size(-1) / 2)
    scale_factor = boxes.new_tensor(scale_factor).repeat((1, repeat_num))
    return boxes * scale_factor


def get_box_wh(boxes: Tensor) -> Tuple[Tensor, Tensor]:
    """Get the width and height of boxes with type of tensor or box type.

    Args:
        boxes (Tensor or :obj:`BaseBoxes`): boxes with type of tensor
            or box type.

    Returns:
        Tuple[Tensor, Tensor]: the width and height of boxes.
    """
    # Tensor boxes will be treated as horizontal boxes by defaults
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return w, h


@HEADS.register_module()
class RTMDetInsHead(RTMDetHead):
    """Detection Head of RTMDet-Ins.

    Args:
        num_prototypes (int): Number of mask prototype features extracted
            from the mask head. Defaults to 8.
        dyconv_channels (int): Channel of the dynamic conv layers.
            Defaults to 8.
        num_dyconvs (int): Number of the dynamic convolution layers.
            Defaults to 3.
        mask_loss_stride (int): Down sample stride of the masks for loss
            computation. Defaults to 4.
        loss_mask (:obj:`ConfigDict` or dict): Config dict for mask loss.
    """

    def __init__(
        self,
        *args,
        num_prototypes: int = 8,
        dyconv_channels: int = 8,
        num_dyconvs: int = 3,
        mask_loss_stride: int = 4,
        loss_mask=dict(type="DiceLoss", loss_weight=2.0, eps=5e-6, reduction="mean"),
        **kwargs
    ) -> None:
        self.num_prototypes = num_prototypes
        self.num_dyconvs = num_dyconvs
        self.dyconv_channels = dyconv_channels
        self.mask_loss_stride = mask_loss_stride
        super().__init__(*args, **kwargs)
        self.loss_mask = build_loss(loss_mask)
        self.roi_align = RoIAlign(output_size=(28, 28))

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        super()._init_layers()
        # a branch to predict kernels of dynamic convs
        self.kernel_convs = nn.ModuleList()
        # calculate num dynamic parameters
        weight_nums, bias_nums = [], []
        for i in range(self.num_dyconvs):
            if i == 0:
                weight_nums.append(
                    # mask prototype and coordinate features
                    (self.num_prototypes + 2)
                    * self.dyconv_channels
                )
                bias_nums.append(self.dyconv_channels * 1)
            elif i == self.num_dyconvs - 1:
                weight_nums.append(self.dyconv_channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.dyconv_channels * self.dyconv_channels)
                bias_nums.append(self.dyconv_channels * 1)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.kernel_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                )
            )
        pred_pad_size = self.pred_kernel_size // 2
        self.rtm_kernel = nn.Conv2d(
            self.feat_channels, self.num_gen_params, self.pred_kernel_size, padding=pred_pad_size
        )
        self.mask_head = MaskFeatModule(
            in_channels=self.in_channels,
            feat_channels=self.feat_channels,
            stacked_convs=4,
            num_levels=len(self.prior_generator.strides),
            num_prototypes=self.num_prototypes,
            act_cfg=self.act_cfg,
            norm_cfg=self.norm_cfg,
        )

    def forward(self, feats: Tuple[Tensor, ...]) -> tuple:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
            - cls_scores (list[Tensor]): Classification scores for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * 4.
            - kernel_preds (list[Tensor]): Dynamic conv kernels for all scale
              levels, each is a 4D-tensor, the channels number is
              num_gen_params.
            - mask_feat (Tensor): Output feature of the mask head. Each is a
              4D-tensor, the channels number is num_prototypes.
        """
        mask_feat = self.mask_head(feats)

        cls_scores = []
        bbox_preds = []
        kernel_preds = []
        for idx, (x, scale, stride) in enumerate(zip(feats, self.scales, self.prior_generator.strides)):
            cls_feat = x
            reg_feat = x
            kernel_feat = x

            for cls_layer in self.cls_convs:
                cls_feat = cls_layer(cls_feat)
            cls_score = self.rtm_cls(cls_feat)

            for kernel_layer in self.kernel_convs:
                kernel_feat = kernel_layer(kernel_feat)
            kernel_pred = self.rtm_kernel(kernel_feat)

            for reg_layer in self.reg_convs:
                reg_feat = reg_layer(reg_feat)

            if self.with_objectness:
                objectness = self.rtm_obj(reg_feat)
                cls_score = inverse_sigmoid(sigmoid_geometric_mean(cls_score, objectness))

            reg_dist = scale(self.rtm_reg(reg_feat)) * stride[0]

            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
            kernel_preds.append(kernel_pred)
        return tuple(cls_scores), tuple(bbox_preds), tuple(kernel_preds), mask_feat

    def get_bboxes(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        kernel_preds: List[Tensor],
        mask_feat: Tensor,
        score_factors: Optional[List[Tensor]] = None,
        img_metas: Optional[List[Dict]] = None,
        cfg: Dict = None,
        rescale: bool = False,
        with_nms: bool = True,
    ) -> List[InstanceData]:
        """Transform a batch of output features extracted from the head into bbox results.

        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.

        Args:
            cls_scores (List[Tensor]): Classification scores for all scale levels, each is a 4D-tensor,
                                       has shape (batch_size, num_priors * num_classes, H, W).
            bbox_preds (List[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            kernel_preds (List[Tensor]): Kernel predictions of dynamic convs for all scale levels,
                                         each is a 4D-tensor, has shape (batch_size, num_params, H, W).
            mask_feat (Tensor): Mask prototype features extracted from the mask head, has shape
                                (batch_size, num_prototypes, H, W).
            score_factors (List[Tensor]): Score factor for all scale level, each is a 4D-tensor, has shape
                                          (batch_size, num_priors * 1, H, W). Defaults to None.
            img_metas (List[Dict]): Batch image meta info. Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, h, w).
        """
        assert len(cls_scores) == len(bbox_preds)

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes, dtype=cls_scores[0].dtype, device=cls_scores[0].device, with_stride=True
        )

        result_list = []

        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id, detach=True)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id, detach=True)
            kernel_pred_list = select_single_mlvl(kernel_preds, img_id, detach=True)
            if with_score_factors:
                score_factor_list = select_single_mlvl(score_factors, img_id, detach=True)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                kernel_pred_list=kernel_pred_list,
                mask_feat=mask_feat[img_id],
                score_factor_list=score_factor_list,
                mlvl_priors=mlvl_priors,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms,
            )
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(
        self,
        cls_score_list: List[Tensor],
        bbox_pred_list: List[Tensor],
        kernel_pred_list: List[Tensor],
        mask_feat: Tensor,
        score_factor_list: List[Tensor],
        mlvl_priors: List[Tensor],
        img_meta: dict,
        cfg: dict,
        rescale: bool = False,
        with_nms: bool = True,
    ) -> InstanceData:
        """Transform a single image's features extracted from the head into bbox and mask results.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            kernel_pred_list (list[Tensor]): Kernel predictions of dynamic
                convs for all scale levels of a single image, each is a
                4D-tensor, has shape (num_params, H, W).
            mask_feat (Tensor): Mask prototype features of a single image
                extracted from the mask head, has shape (num_prototypes, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmengine.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, h, w).
        """
        if score_factor_list[0] is None:
            # e.g. Retina, FreeAnchor, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta["img_shape"]
        nms_pre = cfg.get("nms_pre", -1)

        mlvl_bbox_preds = []
        mlvl_kernels = []
        mlvl_valid_priors = []
        mlvl_scores = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors: List = []
        else:
            mlvl_score_factors = None

        for level_idx, (cls_score, bbox_pred, kernel_pred, score_factor, priors) in enumerate(
            zip(cls_score_list, bbox_pred_list, kernel_pred_list, score_factor_list, mlvl_priors)
        ):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2, 0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            kernel_pred = kernel_pred.permute(1, 2, 0).reshape(-1, self.num_gen_params)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            score_thr = cfg.get("score_thr", 0)

            results = filter_scores_and_topk(
                scores, score_thr, nms_pre, dict(bbox_pred=bbox_pred, priors=priors, kernel_pred=kernel_pred)
            )
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results["bbox_pred"]
            priors = filtered_results["priors"]
            kernel_pred = filtered_results["kernel_pred"]

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            mlvl_kernels.append(kernel_pred)

            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = torch.cat(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors[..., :2], bbox_pred, max_shape=img_shape)

        results = InstanceData()
        results.bboxes = bboxes
        results.priors = priors
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)
        results.kernels = torch.cat(mlvl_kernels)
        if with_score_factors:
            results.score_factors = torch.cat(mlvl_score_factors)

        return self._bbox_mask_post_process(
            results=results, mask_feat=mask_feat, cfg=cfg, rescale=rescale, with_nms=with_nms, img_meta=img_meta
        )

    def _bbox_mask_post_process(
        self,
        results: InstanceData,
        mask_feat,
        cfg: ConfigDict,
        rescale: bool = False,
        with_nms: bool = True,
        img_meta: Optional[dict] = None,
    ) -> InstanceData:
        """Bbox and mask post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            results (:obj:`InstaceData`): Detection instance results,
                each item has shape (num_bboxes, ).
            mask_feat (Tensor): Mask prototype features of a single image
            cfg (ConfigDict): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default to False.
            with_nms (bool): If True, do nms before return boxes.
                Default to True.
            img_meta (dict, optional): Image meta info. Defaults to None.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, h, w).
        """
        stride = self.prior_generator.strides[0][0]
        if rescale:
            assert img_meta.get("scale_factor") is not None
            scale_factor = [1 / s for s in img_meta["scale_factor"]]
            scale_factor = results.bboxes.new_tensor(scale_factor)
            results.bboxes = results.bboxes * scale_factor

        if hasattr(results, "score_factors"):
            # TODO: Add sqrt operation in order to be consistent with
            #  the paper.
            score_factors = results.pop("score_factors")
            results.scores = results.scores * score_factors

        # filter small size bboxes
        if cfg.get("min_bbox_size", -1) >= 0:
            w, h = get_box_wh(results.bboxes)
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]

        # TODO: deal with `with_nms` and `nms_cfg=None` in test_cfg
        assert with_nms, "with_nms must be True for RTMDet-Ins"
        if results.bboxes.numel() > 0:
            bboxes = results.bboxes
            det_bboxes, keep_idxs = batched_nms(bboxes, results.scores, results.labels, cfg.nms)
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
            results = results[: cfg.max_per_img]

            # process masks
            mask_logits = self._mask_predict_by_feat_single(mask_feat, results.kernels, results.priors)
            mask_logits = F.interpolate(mask_logits.unsqueeze(0), scale_factor=stride, mode="bilinear")
            masks = mask_logits.sigmoid()

            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            _bboxes = results.bboxes / scale_factor if rescale else results.bboxes[:, :4]
            rois = bbox2roi([_bboxes])
            cropped_masks = self.roi_align(masks, rois)
            cropped_masks = cropped_masks[torch.arange(cropped_masks.size(0)), torch.arange(cropped_masks.size(0))]
            cropped_masks = cropped_masks > cfg.mask_thr_binary
            results.masks = cropped_masks
        else:
            h, w = img_meta["ori_shape"][:2] if rescale else img_meta["img_shape"][:2]
            results.masks = torch.zeros(
                size=(results.bboxes.shape[0], h, w), dtype=torch.bool, device=results.bboxes.device
            )

        return results

    def parse_dynamic_params(self, flatten_kernels: Tensor) -> tuple:
        """Split kernel head prediction to conv weight and bias."""
        n_inst = flatten_kernels.size(0)
        n_layers = len(self.weight_nums)
        params_splits = list(torch.split_with_sizes(flatten_kernels, self.weight_nums + self.bias_nums, dim=1))
        weight_splits = params_splits[:n_layers]
        bias_splits = params_splits[n_layers:]
        for i in range(n_layers):
            if i < n_layers - 1:
                weight_splits[i] = weight_splits[i].reshape(n_inst * self.dyconv_channels, -1, 1, 1)
                bias_splits[i] = bias_splits[i].reshape(n_inst * self.dyconv_channels)
            else:
                weight_splits[i] = weight_splits[i].reshape(n_inst, -1, 1, 1)
                bias_splits[i] = bias_splits[i].reshape(n_inst)

        return weight_splits, bias_splits

    def _mask_predict_by_feat_single(self, mask_feat: Tensor, kernels: Tensor, priors: Tensor) -> Tensor:
        """Generate mask logits from mask features with dynamic convs.

        Args:
            mask_feat (Tensor): Mask prototype features.
                Has shape (num_prototypes, H, W).
            kernels (Tensor): Kernel parameters for each instance.
                Has shape (num_instance, num_params)
            priors (Tensor): Center priors for each instance.
                Has shape (num_instance, 4).


        Returns:
            Tensor: Instance segmentation masks for each instance.
                Has shape (num_instance, H, W).
        """
        num_inst = priors.shape[0]
        h, w = mask_feat.size()[-2:]
        if num_inst < 1:
            return torch.empty(size=(num_inst, h, w), dtype=mask_feat.dtype, device=mask_feat.device)
        if len(mask_feat.shape) < 4:
            mask_feat.unsqueeze(0)

        coord = self.prior_generator.single_level_grid_priors((h, w), level_idx=0, device=mask_feat.device).reshape(
            1, -1, 2
        )
        num_inst = priors.shape[0]
        points = priors[:, :2].reshape(-1, 1, 2)
        strides = priors[:, 2:].reshape(-1, 1, 2)
        relative_coord = (points - coord).permute(0, 2, 1) / (strides[..., 0].reshape(-1, 1, 1) * 8)
        relative_coord = relative_coord.reshape(num_inst, 2, h, w)

        mask_feat = torch.cat([relative_coord, mask_feat.repeat(num_inst, 1, 1, 1)], dim=1)
        weights, biases = self.parse_dynamic_params(kernels)

        n_layers = len(weights)
        x = mask_feat.reshape(1, -1, h, w)
        for i, (weight, bias) in enumerate(zip(weights, biases)):
            x = F.conv2d(x, weight, bias=bias, stride=1, padding=0, groups=num_inst)
            if i < n_layers - 1:
                x = F.relu(x)
        x = x.reshape(num_inst, h, w)
        return x

    def loss_mask_by_feat(
        self,
        mask_feats: Tensor,
        flatten_kernels: Tensor,
        sampling_results_list: List[SamplingResult],
        batch_gt_instances: List[InstanceData],
    ) -> Tensor:
        """Compute instance segmentation loss.

        Args:
            mask_feats (Tensor): Mask prototype features extracted from the mask head.
                                 Has shape (N, num_prototypes, H, W).
            flatten_kernels (Tensor): Kernels of the dynamic conv layers.
                                      Has shape (N, num_instances, num_params).
            sampling_results_list (List[SamplingResult]): Batch of assignment results.
            batch_gt_instances (List[InstanceData]): Batch of gt_instance.

        Returns:
            Tensor: The mask loss tensor.
        """
        batch_pos_mask_logits = []
        pos_gt_masks = []
        for idx, (mask_feat, kernels, sampling_results, gt_instances) in enumerate(
            zip(mask_feats, flatten_kernels, sampling_results_list, batch_gt_instances)
        ):
            pos_priors = sampling_results.pos_bboxes
            pos_inds = sampling_results.pos_inds
            pos_kernels = kernels[pos_inds]  # n_pos, num_gen_params
            pos_mask_logits = self._mask_predict_by_feat_single(mask_feat, pos_kernels, pos_priors)
            if gt_instances.masks.numel() == 0:
                gt_masks = torch.empty_like(gt_instances.masks)
            else:
                gt_masks = gt_instances.masks[sampling_results.pos_assigned_gt_inds, :]
            batch_pos_mask_logits.append(pos_mask_logits)
            pos_gt_masks.append(gt_masks)

        pos_gt_masks = torch.cat(pos_gt_masks, 0)
        batch_pos_mask_logits = torch.cat(batch_pos_mask_logits, 0)

        # avg_factor
        num_pos = batch_pos_mask_logits.shape[0]  # type: ignore[attr-defined]
        num_pos = reduce_mean(mask_feats.new_tensor([num_pos])).clamp_(min=1).item()

        if batch_pos_mask_logits.shape[0] == 0:  # type: ignore[attr-defined]
            return mask_feats.sum() * 0

        scale = self.prior_generator.strides[0][0] // self.mask_loss_stride
        # upsample pred masks
        batch_pos_mask_logits = batch_pos_mask_logits.unsqueeze(0)  # type: ignore[attr-defined]
        batch_pos_mask_logits = F.interpolate(
            batch_pos_mask_logits, scale_factor=scale, mode="bilinear", align_corners=False
        ).squeeze(
            0
        )  # type: ignore[attr-defined]
        # downsample gt masks
        pos_gt_masks = pos_gt_masks[
            :, self.mask_loss_stride // 2 :: self.mask_loss_stride, self.mask_loss_stride // 2 :: self.mask_loss_stride
        ]  # type: ignore[call-overload]

        loss_mask = self.loss_mask(batch_pos_mask_logits, pos_gt_masks, weight=None, avg_factor=num_pos)

        return loss_mask

    def loss(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        kernel_preds: List[Tensor],
        mask_feat: Tensor,
        batch_gt_instances: List[InstanceData],
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: List[InstanceData] = None,
    ):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Decoded box for each scale
                level with shape (N, num_anchors * 4, H, W) in
                [tl_x, tl_y, br_x, br_y] format.
            kernel_preds (list[Tensor]): Kernel predictions of dynamic.
            mask_feat (Tensor): Mask prototype features.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_imgs = len(batch_img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, batch_img_metas, device=device)
        flatten_cls_scores = torch.cat(
            [cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels) for cls_score in cls_scores], 1
        )
        flatten_kernels = torch.cat(
            [
                kernel_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_gen_params)
                for kernel_pred in kernel_preds
            ],
            1,
        )
        decoded_bboxes = []
        for anchor, bbox_pred in zip(anchor_list[0], bbox_preds):
            anchor = anchor.reshape(-1, 4)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            bbox_pred = distance2bbox(anchor, bbox_pred)
            decoded_bboxes.append(bbox_pred)

        flatten_bboxes = torch.cat(decoded_bboxes, 1)

        cls_reg_targets = self.get_targets(
            flatten_cls_scores,
            flatten_bboxes,
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
        )
        (
            anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            assign_metrics_list,
            sampling_results_list,
        ) = cls_reg_targets

        losses_cls, losses_bbox, cls_avg_factors, bbox_avg_factors = multi_apply(
            self.loss_by_feat_single,
            cls_scores,
            decoded_bboxes,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            assign_metrics_list,
            self.prior_generator.strides,
        )

        cls_avg_factor = reduce_mean(sum(cls_avg_factors)).clamp_(min=1).item()
        losses_cls = list(map(lambda x: x / cls_avg_factor, losses_cls))

        bbox_avg_factor = reduce_mean(sum(bbox_avg_factors)).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))

        loss_mask = self.loss_mask_by_feat(mask_feat, flatten_kernels, sampling_results_list, batch_gt_instances)
        loss = dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_mask=loss_mask)
        return loss


class MaskFeatModule(BaseModule):
    """Mask feature head used in RTMDet-Ins.

    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels of the mask feature
             map branch.
        num_levels (int): The starting feature map level from RPN that
             will be used to predict the mask feature map.
        num_prototypes (int): Number of output channel of the mask feature
             map branch. This is the channel count of the mask
             feature map that to be dynamically convolved with the predicted
             kernel.
        stacked_convs (int): Number of convs in mask feature branch.
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Default: dict(type='ReLU', inplace=True)
        norm_cfg (dict): Config dict for normalization layer. Default: None.
    """

    def __init__(
        self,
        in_channels: int,
        feat_channels: int = 256,
        stacked_convs: int = 4,
        num_levels: int = 3,
        num_prototypes: int = 8,
        act_cfg: dict = dict(type="ReLU", inplace=True),
        norm_cfg: dict = dict(type="BN"),
    ) -> None:
        super().__init__(init_cfg=None)
        self.num_levels = num_levels
        self.fusion_conv = nn.Conv2d(num_levels * in_channels, in_channels, 1)
        convs = []
        for i in range(stacked_convs):
            in_c = in_channels if i == 0 else feat_channels
            convs.append(ConvModule(in_c, feat_channels, 3, padding=1, act_cfg=act_cfg, norm_cfg=norm_cfg))
        self.stacked_convs = nn.Sequential(*convs)
        self.projection = nn.Conv2d(feat_channels, num_prototypes, kernel_size=1)

    def forward(self, features: Tuple[Tensor, ...]) -> Tensor:
        """Forward features from the upstream network."""
        # multi-level feature fusion
        fusion_feats = [features[0]]
        size = features[0].shape[-2:]
        for i in range(1, self.num_levels):
            f = F.interpolate(features[i], size=size, mode="bilinear")
            fusion_feats.append(f)
        fusion_feats = torch.cat(fusion_feats, dim=1)
        fusion_feats = self.fusion_conv(fusion_feats)
        # pred mask feats
        mask_features = self.stacked_convs(fusion_feats)
        mask_features = self.projection(mask_features)
        return mask_features


@HEADS.register_module()
class RTMDetInsSepBNHead(RTMDetInsHead):
    """Detection Head of RTMDet-Ins with sep-bn layers.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        share_conv (bool): Whether to share conv layers between stages.
            Defaults to True.
        norm_cfg (:obj:`ConfigDict` or dict)): Config dict for normalization
            layer. Defaults to dict(type='BN').
        act_cfg (:obj:`ConfigDict` or dict)): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        pred_kernel_size (int): Kernel size of prediction layer. Defaults to 1.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        share_conv: bool = True,
        with_objectness: bool = False,
        norm_cfg: dict = dict(type="BN", requires_grad=True),
        act_cfg: dict = dict(type="SiLU", inplace=True),
        pred_kernel_size: int = 1,
        **kwargs
    ) -> None:
        self.share_conv = share_conv
        super().__init__(
            num_classes,
            in_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            pred_kernel_size=pred_kernel_size,
            with_objectness=with_objectness,
            **kwargs
        )

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.kernel_convs = nn.ModuleList()

        self.rtm_cls = nn.ModuleList()
        self.rtm_reg = nn.ModuleList()
        self.rtm_kernel = nn.ModuleList()
        self.rtm_obj = nn.ModuleList()

        # calculate num dynamic parameters
        weight_nums, bias_nums = [], []
        for i in range(self.num_dyconvs):
            if i == 0:
                weight_nums.append((self.num_prototypes + 2) * self.dyconv_channels)
                bias_nums.append(self.dyconv_channels)
            elif i == self.num_dyconvs - 1:
                weight_nums.append(self.dyconv_channels)
                bias_nums.append(1)
            else:
                weight_nums.append(self.dyconv_channels * self.dyconv_channels)
                bias_nums.append(self.dyconv_channels)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)
        pred_pad_size = self.pred_kernel_size // 2

        for n in range(len(self.prior_generator.strides)):
            cls_convs = nn.ModuleList()
            reg_convs = nn.ModuleList()
            kernel_convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                cls_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                    )
                )
                reg_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                    )
                )
                kernel_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                    )
                )
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(cls_convs)
            self.kernel_convs.append(kernel_convs)

            self.rtm_cls.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * self.cls_out_channels,
                    self.pred_kernel_size,
                    padding=pred_pad_size,
                )
            )
            self.rtm_reg.append(
                nn.Conv2d(self.feat_channels, self.num_base_priors * 4, self.pred_kernel_size, padding=pred_pad_size)
            )
            self.rtm_kernel.append(
                nn.Conv2d(self.feat_channels, self.num_gen_params, self.pred_kernel_size, padding=pred_pad_size)
            )
            if self.with_objectness:
                self.rtm_obj.append(nn.Conv2d(self.feat_channels, 1, self.pred_kernel_size, padding=pred_pad_size))

        if self.share_conv:
            for n in range(len(self.prior_generator.strides)):
                for i in range(self.stacked_convs):
                    self.cls_convs[n][i].conv = self.cls_convs[0][i].conv
                    self.reg_convs[n][i].conv = self.reg_convs[0][i].conv

        self.mask_head = MaskFeatModule(
            in_channels=self.in_channels,
            feat_channels=self.feat_channels,
            stacked_convs=4,
            num_levels=len(self.prior_generator.strides),
            num_prototypes=self.num_prototypes,
            act_cfg=self.act_cfg,
            norm_cfg=self.norm_cfg,
        )

    def init_weights(self) -> None:
        """Initialize weights of the head."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)
        bias_cls = bias_init_with_prob(0.01)
        for rtm_cls, rtm_reg, rtm_kernel in zip(self.rtm_cls, self.rtm_reg, self.rtm_kernel):
            normal_init(rtm_cls, std=0.01, bias=bias_cls)
            normal_init(rtm_reg, std=0.01, bias=1)
        if self.with_objectness:
            for rtm_obj in self.rtm_obj:
                normal_init(rtm_obj, std=0.01, bias=bias_cls)

    def forward(self, feats: Tuple[Tensor, ...]) -> tuple:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
            - cls_scores (list[Tensor]): Classification scores for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * 4.
            - kernel_preds (list[Tensor]): Dynamic conv kernels for all scale
              levels, each is a 4D-tensor, the channels number is
              num_gen_params.
            - mask_feat (Tensor): Output feature of the mask head. Each is a
              4D-tensor, the channels number is num_prototypes.
        """
        mask_feat = self.mask_head(feats)

        cls_scores = []
        bbox_preds = []
        kernel_preds = []
        for idx, (x, stride) in enumerate(zip(feats, self.prior_generator.strides)):
            cls_feat = x
            reg_feat = x
            kernel_feat = x

            for cls_layer in self.cls_convs[idx]:
                cls_feat = cls_layer(cls_feat)
            cls_score = self.rtm_cls[idx](cls_feat)

            for kernel_layer in self.kernel_convs[idx]:
                kernel_feat = kernel_layer(kernel_feat)
            kernel_pred = self.rtm_kernel[idx](kernel_feat)

            for reg_layer in self.reg_convs[idx]:
                reg_feat = reg_layer(reg_feat)

            if self.with_objectness:
                objectness = self.rtm_obj[idx](reg_feat)
                cls_score = inverse_sigmoid(sigmoid_geometric_mean(cls_score, objectness))

            reg_dist = F.relu(self.rtm_reg[idx](reg_feat)) * stride[0]

            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
            kernel_preds.append(kernel_pred)
        return tuple(cls_scores), tuple(bbox_preds), tuple(kernel_preds), mask_feat


if is_mmdeploy_enabled():

    from typing import List, Optional

    from mmdeploy.codebase.mmdet import get_post_processing_params
    from mmdeploy.core import FUNCTION_REWRITER
    from mmdeploy.mmcv.ops import ONNXNMSop
    from mmengine.config import ConfigDict
    from packaging import version
    from torch import Tensor

    @FUNCTION_REWRITER.register_rewriter(
        func_name="otx.algorithms.detection.adapters.mmdet.models.dense_heads."
        "rtmdet_ins_head.RTMDetInsHead.get_bboxes"
    )
    def rtmdet_ins_head__get_bboxes(
        ctx,
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        kernel_preds: List[Tensor],
        mask_feat: Tensor,
        img_metas: Optional[List[Dict]] = None,
        cfg: Optional[ConfigDict] = None,
        rescale: bool = False,
    ):
        """Rewrite `get_bboxes` of `RTMDet-Ins` for default backend.

        Rewrite this function to deploy model, transform network output for a
        batch into bbox predictions.

        Args:
            ctx: Context that contains original meta information.
            self: The instance of `RTMDetInsHead`.
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            kernel_preds (list[Tensor]): Dynamic conv kernels.
            mask_feat (Tensor): Output feature of the mask head.
            img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.


        Returns:
            tuple[Tensor, Tensor]: The first item is an (N, num_box, 5) tensor,
                where 5 represent (tl_x, tl_y, br_x, br_y, score), N is batch
                size and the score between 0 and 1. The shape of the second
                tensor in the tuple is (N, num_box), and each element
                represents the class label of the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)
        device = cls_scores[0].device
        cfg = self.test_cfg if cfg is None else cfg
        batch_size = bbox_preds[0].shape[0]
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(featmap_sizes, device=device, with_stride=True)

        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1, self.cls_out_channels) for cls_score in cls_scores
        ]
        flatten_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4) for bbox_pred in bbox_preds]
        flatten_kernel_preds = [
            kernel_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_gen_params) for kernel_pred in kernel_preds
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_kernel_preds = torch.cat(flatten_kernel_preds, dim=1)
        priors = torch.cat(mlvl_priors)
        tl_x = priors[..., 0] - flatten_bbox_preds[..., 0]  # type: ignore[call-overload]
        tl_y = priors[..., 1] - flatten_bbox_preds[..., 1]  # type: ignore[call-overload]
        br_x = priors[..., 0] + flatten_bbox_preds[..., 2]  # type: ignore[call-overload]
        br_y = priors[..., 1] + flatten_bbox_preds[..., 3]  # type: ignore[call-overload]
        bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
        # directly multiply score factor and feed to nms
        max_scores, _ = torch.max(flatten_cls_scores, 1)
        mask = max_scores >= cfg.score_thr
        scores = flatten_cls_scores.where(mask, flatten_cls_scores.new_zeros(1))  # type: ignore[attr-defined]

        deploy_cfg = ctx.cfg
        post_params = get_post_processing_params(deploy_cfg)
        max_output_boxes_per_class = post_params.max_output_boxes_per_class
        iou_threshold = cfg.nms.get("iou_threshold", post_params.iou_threshold)
        score_threshold = cfg.get("score_thr", post_params.score_threshold)
        pre_top_k = post_params.pre_top_k
        keep_top_k = cfg.get("max_per_img", post_params.keep_top_k)
        mask_thr_binary = cfg.get("mask_thr_binary", 0.5)

        return _nms_with_mask_static(
            self,
            priors,
            bboxes,
            scores,
            flatten_kernel_preds,
            mask_feat,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
            pre_top_k,
            keep_top_k,
            mask_thr_binary,
        )

    def _nms_with_mask_static(
        self,
        priors: Tensor,
        boxes: Tensor,
        scores: Tensor,
        kernels: Tensor,
        mask_feats: Tensor,
        max_output_boxes_per_class: int = 1000,
        iou_threshold: float = 0.5,
        score_threshold: float = 0.05,
        pre_top_k: int = -1,
        keep_top_k: int = -1,
        mask_thr_binary: float = 0.5,
    ):
        """Wrapper for `multiclass_nms` with ONNXRuntime.

        Args:
            self: The instance of `RTMDetInsHead`.
            priors (Tensor): The prior boxes of shape [num_boxes, 4].
            boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
            scores (Tensor): The detection scores of shape
                [N, num_boxes, num_classes].
            kernels (Tensor): The dynamic conv kernels.
            mask_feats (Tensor): The mask feature.
            max_output_boxes_per_class (int): Maximum number of output
                boxes per class of nms. Defaults to 1000.
            iou_threshold (float): IOU threshold of nms. Defaults to 0.5.
            score_threshold (float): score threshold of nms.
                Defaults to 0.05.
            pre_top_k (int): Number of top K boxes to keep before nms.
                Defaults to -1.
            keep_top_k (int): Number of top K boxes to keep after nms.
                Defaults to -1.
            mask_thr_binary (float): Binarization threshold for masks.


        Returns:
            tuple[Tensor, Tensor]: (dets, labels), `dets` of shape [N, num_det, 5]
                and `labels` of shape [N, num_det].
        """
        if version.parse(torch.__version__) < version.parse("1.13.0"):
            max_output_boxes_per_class = torch.LongTensor([max_output_boxes_per_class])
        iou_threshold = torch.tensor([iou_threshold], dtype=torch.float32)
        score_threshold = torch.tensor([score_threshold], dtype=torch.float32)

        # pre topk
        if pre_top_k > 0:
            max_scores, _ = scores.max(-1)
            _, topk_inds = max_scores.squeeze(0).topk(pre_top_k)
            boxes = boxes[:, topk_inds, :]
            scores = scores[:, topk_inds, :]
            kernels = kernels[:, topk_inds, :]
            priors = priors[topk_inds, :]

        scores = scores.permute(0, 2, 1)
        selected_indices = ONNXNMSop.apply(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)

        cls_inds = selected_indices[:, 1]
        box_inds = selected_indices[:, 2]

        scores = scores[:, cls_inds, box_inds].unsqueeze(2)
        boxes = boxes[:, box_inds, ...]
        kernels = kernels[:, box_inds, :]
        priors = priors[box_inds, :]
        dets = torch.cat([boxes, scores], dim=2)
        labels = cls_inds.unsqueeze(0)

        # pad
        dets = torch.cat((dets, dets.new_zeros((1, 1, 5))), 1)
        labels = torch.cat((labels, labels.new_zeros((1, 1))), 1)
        kernels = torch.cat((kernels, kernels.new_zeros(1, 1, kernels.shape[2])), 1)
        priors = torch.cat((priors, priors.new_zeros(1, 4)), 0)

        # topk or sort
        is_use_topk = keep_top_k > 0 and (torch.onnx.is_in_onnx_export() or keep_top_k < dets.shape[1])
        if is_use_topk:
            _, topk_inds = dets[:, :, -1].topk(keep_top_k, dim=1)
        else:
            _, topk_inds = dets[:, :, -1].sort(dim=1, descending=True)
        topk_inds = topk_inds.squeeze(0)
        dets = dets[:, topk_inds, ...]
        labels = labels[:, topk_inds, ...]
        kernels = kernels[:, topk_inds, ...]
        priors = priors[topk_inds, ...]
        mask_logits = _mask_predict_by_feat_single(self, mask_feats, kernels[0], priors)
        stride = self.prior_generator.strides[0][0]
        mask_logits = F.interpolate(mask_logits.unsqueeze(0), scale_factor=stride, mode="bilinear")
        masks = mask_logits.sigmoid()

        batch_index = (
            torch.arange(dets.size(0), device=dets.device).float().view(-1, 1, 1).expand(dets.size(0), dets.size(1), 1)
        )
        rois = torch.cat([batch_index, dets[..., :4]], dim=-1)
        cropped_masks = self.roi_align(masks, rois[0])
        cropped_masks = cropped_masks[torch.arange(cropped_masks.size(0)), torch.arange(cropped_masks.size(0))]
        cropped_masks = cropped_masks.unsqueeze(0)
        return dets, labels, cropped_masks

    def _mask_predict_by_feat_single(self, mask_feat, kernels, priors):
        """Decode mask with dynamic conv."""
        num_inst = priors.shape[0]
        h, w = mask_feat.size()[-2:]
        if num_inst < 1:
            return torch.empty(size=(num_inst, h, w), dtype=mask_feat.dtype, device=mask_feat.device)
        if len(mask_feat.shape) < 4:
            mask_feat.unsqueeze(0)
        coord = (
            self.prior_generator.single_level_grid_priors((h, w), level_idx=0).reshape(1, -1, 2).to(mask_feat.device)
        )
        num_inst = priors.shape[0]
        points = priors[:, :2].reshape(-1, 1, 2)
        strides = priors[:, 2:].reshape(-1, 1, 2)
        relative_coord = (points - coord).permute(0, 2, 1) / (strides[..., 0].reshape(-1, 1, 1) * 8)
        relative_coord = relative_coord.reshape(num_inst, 2, h, w)

        mask_feat = torch.cat([relative_coord, mask_feat.repeat(num_inst, 1, 1, 1)], dim=1)
        weights, biases = _parse_dynamic_params(self, kernels)

        n_layers = len(weights)
        x = mask_feat.flatten(2)
        for i, (weight, bias) in enumerate(zip(weights, biases)):
            # replace dynamic conv with bmm
            x = torch.bmm(weight, x)
            x = x + bias[:, :, None]
            if i < n_layers - 1:
                x = x.clamp_(min=0)
        x = x.reshape(num_inst, h, w)
        return x

    def _parse_dynamic_params(self, flatten_kernels):
        """Split kernel head prediction to conv weight and bias."""
        n_inst = flatten_kernels.size(0)
        n_layers = len(self.weight_nums)
        params_splits = list(torch.split_with_sizes(flatten_kernels, self.weight_nums + self.bias_nums, dim=1))
        weight_splits = params_splits[:n_layers]
        bias_splits = params_splits[n_layers:]
        for idx in range(n_layers):
            if idx < n_layers - 1:
                weight_splits[idx] = weight_splits[idx].reshape(n_inst, self.dyconv_channels, -1)
            else:
                weight_splits[idx] = weight_splits[idx].reshape(n_inst, 1, -1)
        return weight_splits, bias_splits
