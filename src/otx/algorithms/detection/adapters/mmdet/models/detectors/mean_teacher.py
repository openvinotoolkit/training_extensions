"""UnbiasedTeacher Class for mmdetection detectors."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import copy
import functools

import cv2
import mmcv
import numpy as np
import torch
from mmdet.core import bbox2result, bbox2roi
from mmdet.core.mask.structures import BitmapMasks
from mmdet.models import DETECTORS, build_detector
from mmdet.models.detectors import BaseDetector
from torch import distributed as dist

from otx.utils.logger import get_logger

from .sam_detector_mixin import SAMDetectorMixin

logger = get_logger()

# TODO: Need to fix pylint issues
# pylint: disable=abstract-method, too-many-locals, unused-argument


@DETECTORS.register_module()
class MeanTeacher(SAMDetectorMixin, BaseDetector):
    """Mean teacher framework for detection and instance segmentation."""

    def __init__(
        self,
        arch_type,
        unlabeled_loss_weights={"cls": 1.0, "bbox": 1.0, "mask": 1.0},
        pseudo_conf_thresh=0.7,
        bg_loss_weight=-1.0,
        min_pseudo_label_ratio=0.0,
        visualize=False,
        filter_empty_annotations=False,
        **kwargs,
    ):
        super().__init__()
        self.unlabeled_loss_weights = unlabeled_loss_weights
        self.pseudo_conf_thresh = pseudo_conf_thresh
        self.bg_loss_weight = bg_loss_weight
        self.min_pseudo_label_ratio = min_pseudo_label_ratio
        self.filter_empty_annotations = filter_empty_annotations
        self.visualize = visualize
        cfg = kwargs.copy()
        cfg["type"] = arch_type
        self.model_s = build_detector(cfg)
        self.model_t = copy.deepcopy(self.model_s)
        self.model_t.eval()
        # warmup for first epochs
        self.enable_unlabeled_loss(False)
        self.cur_iter = 0

        # Hooks for super_type transparent weight load/save
        self._register_state_dict_hook(self.state_dict_hook)
        self._register_load_state_dict_pre_hook(functools.partial(self.load_state_dict_pre_hook, self))

    def extract_feat(self, imgs):
        """Extract features for UnbiasedTeacher."""
        return self.model_t.extract_feat(imgs)

    def simple_test(self, img, img_metas, **kwargs):
        """Test from img with UnbiasedTeacher."""
        return self.model_t.simple_test(img, img_metas, **kwargs)

    def aug_test(self, imgs, img_metas, **kwargs):
        """Aug Test from img with UnbiasedTeacher."""
        return self.model_t.aug_test(imgs, img_metas, **kwargs)

    def forward_dummy(self, img, **kwargs):
        """Dummy forward function for UnbiasedTeacher."""
        return self.model_t.forward_dummy(img, **kwargs)

    def enable_unlabeled_loss(self, mode=True):
        """Enable function for UnbiasedTeacher unlabeled loss."""
        self.unlabeled_loss_enabled = mode

    def forward_teacher(self, img, img_metas):
        """Method to extract predictions (pseudo labeles) from teacher."""
        x = self.model_t.extract_feat(img)
        proposal_list = self.model_t.rpn_head.simple_test_rpn(x, img_metas)

        det_bboxes, det_labels = self.model_t.roi_head.simple_test_bboxes(
            x, img_metas, proposal_list, self.model_t.test_cfg.rcnn, rescale=False
        )

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i], self.model_t.roi_head.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.model_t.with_mask:
            return bbox_results
        else:
            ori_shapes = tuple(meta["ori_shape"] for meta in img_metas)
            scale_factors = tuple(meta["scale_factor"] for meta in img_metas)

            num_imgs = len(det_bboxes)
            if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
                segm_results = [
                    [[] for _ in range(self.model_t.roi_head.mask_head.num_classes)] for _ in range(num_imgs)
                ]
            else:
                _bboxes = [det_bboxes[i][:, :4] for i in range(len(det_bboxes))]
                mask_rois = bbox2roi(_bboxes)
                mask_results = self.model_t.roi_head._mask_forward(x, mask_rois)
                mask_pred = mask_results["mask_pred"]
                # split batch mask prediction back to each image
                num_mask_roi_per_img = [len(det_bbox) for det_bbox in det_bboxes]
                mask_preds = mask_pred.split(num_mask_roi_per_img, 0)

                # apply mask post-processing to each image individually
                segm_results = []
                for i in range(num_imgs):
                    if det_bboxes[i].shape[0] == 0:
                        segm_results.append([[] for _ in range(self.model_t.roi_head.mask_head.num_classes)])
                    else:
                        segm_result = self.model_t.roi_head.mask_head.get_scaled_seg_masks(
                            mask_preds[i],
                            _bboxes[i],
                            det_labels[i],
                            self.model_t.test_cfg.rcnn,
                            ori_shapes[i],
                            scale_factors[i],
                            rescale=False,
                        )
                        segm_results.append(segm_result)

        return list(zip(bbox_results, segm_results))

    def forward_train(
        self, img, img_metas, img0=None, gt_bboxes=None, gt_labels=None, gt_masks=None, gt_bboxes_ignore=None, **kwargs
    ):
        """Forward function for UnbiasedTeacher."""
        losses = {}
        self.cur_iter += 1
        # Supervised loss
        # TODO: check img0 only option (which is common for mean teacher method)
        if img0 is not None:
            img = torch.cat((img0, img))  # weak + hard augmented images
            img_metas = img_metas + img_metas
            gt_bboxes = gt_bboxes + gt_bboxes
            gt_labels = gt_labels + gt_labels
            gt_bboxes_ignore = gt_bboxes_ignore + gt_bboxes_ignore if gt_bboxes_ignore else None
            gt_masks = gt_masks + gt_masks if self.model_s.with_mask else None

        forward_train = functools.partial(
            self.model_s.forward_train,
            img,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore if gt_bboxes_ignore else None,
        )
        if self.model_s.with_mask:
            sl_losses = forward_train(gt_masks=gt_masks)
        else:
            sl_losses = forward_train()
        losses.update(sl_losses)

        if not self.unlabeled_loss_enabled:
            return losses

        # Pseudo labels from teacher
        ul_args = kwargs.get("extra_0", {})
        ul_img = ul_args.get("img")
        ul_img0 = ul_args.get("img0")
        ul_img_metas = ul_args.get("img_metas")
        if ul_img is None:
            return losses
        with torch.no_grad():
            if self.model_t.with_mask:
                teacher_outputs = self.forward_teacher(ul_img0, ul_img_metas)
            else:
                teacher_outputs = self.model_t.forward_test([ul_img0], [ul_img_metas], rescale=False)
        current_device = ul_img0[0].device
        pseudo_bboxes, pseudo_labels, pseudo_masks, pseudo_ratio = self.generate_pseudo_labels(
            teacher_outputs, device=current_device, img_meta=ul_img_metas, **kwargs
        )
        non_empty = [bool(len(i)) for i in pseudo_labels] if self.filter_empty_annotations else [True]
        get_unlabeled_loss = pseudo_ratio >= self.min_pseudo_label_ratio and any(non_empty)

        if dist.is_initialized():
            reduced_get_unlabeled_loss = torch.tensor(int(get_unlabeled_loss)).to(current_device)
            dist.all_reduce(reduced_get_unlabeled_loss)
            if dont_have_to_train := not get_unlabeled_loss and reduced_get_unlabeled_loss > 0:
                get_unlabeled_loss = True
                non_empty[0] = True

        losses.update(ps_ratio=torch.tensor([pseudo_ratio], device=current_device))

        # Unsupervised loss
        # Compute only if min_pseudo_label_ratio is reached
        if get_unlabeled_loss:
            if self.filter_empty_annotations:
                pseudo_bboxes = [pb for i, pb in enumerate(pseudo_bboxes) if non_empty[i]]
                pseudo_labels = [pl for i, pl in enumerate(pseudo_labels) if non_empty[i]]
                pseudo_masks = [pm for i, pm in enumerate(pseudo_masks) if non_empty[i]]
                ul_img_metas = [im for i, im in enumerate(ul_img_metas) if non_empty[i]]
                ul_img = ul_img[non_empty]
                if self.visualize:
                    self._visual_online(ul_img, pseudo_bboxes, pseudo_labels)
            if self.bg_loss_weight >= 0.0:
                self.model_s.bbox_head.bg_loss_weight = self.bg_loss_weight
            if self.model_t.with_mask:
                ul_losses = self.model_s.forward_train(
                    ul_img, ul_img_metas, pseudo_bboxes, pseudo_labels, gt_masks=pseudo_masks
                )
            else:
                ul_losses = self.model_s.forward_train(ul_img, ul_img_metas, pseudo_bboxes, pseudo_labels)

            if self.bg_loss_weight >= 0.0:
                self.model_s.bbox_head.bg_loss_weight = -1.0

            for ul_loss_name in ul_losses.keys():
                if ul_loss_name.startswith("loss_"):
                    ul_loss = ul_losses[ul_loss_name]
                    target_loss = ul_loss_name.split("_")[-1]
                    if dist.is_initialized():
                        if dont_have_to_train:
                            self.unlabeled_loss_weights[target_loss] = 0
                    elif self.unlabeled_loss_weights[target_loss] == 0:
                        continue
                    self._update_unlabeled_loss(losses, ul_loss, ul_loss_name, self.unlabeled_loss_weights[target_loss])
        return losses

    def _visual_online(self, img, boxes_list, labels_list, img_id=0, boxes_ignore_list=None, proposal_list=None):
        if not img.size(0):
            return
        img_norm_cfg = dict(mean=np.array([0, 0, 0]), std=np.array([255, 255, 255]))
        img_np = img[img_id].permute(1, 2, 0).cpu().numpy()
        img_np = mmcv.imdenormalize(img_np, **img_norm_cfg)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        boxes, labels = boxes_list[img_id], labels_list[img_id]
        # proposal
        if proposal_list:
            proposal = proposal_list[img_id]
            for idx, box in enumerate(proposal[:, :4]):
                x1, y1, x2, y2 = [int(a.cpu().item()) for a in box]
                img_np = cv2.rectangle(img_np, (x1, y1), (x2, y2), (214, 39, 40), 2)
                cv2.putText(img_np, f"{idx}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (214, 39, 40), 2)
        # ignore
        if boxes_ignore_list:
            boxes_ignore = boxes_ignore_list[img_id]
            for idx, box in enumerate(boxes_ignore):
                x1, y1, x2, y2 = [int(a.cpu().item()) for a in box]
                img_np = cv2.rectangle(img_np, (x1, y1), (x2, y2), (44, 160, 44), 2)
                cv2.putText(img_np, f"{idx}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (44, 160, 44), 2)
        # pseudo gt
        for idx, (box, label) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = [int(a.cpu().item()) for a in box]
            img_np = cv2.rectangle(img_np, (x1, y1), (x2, y2), (157, 80, 136), 2)
            cv2.putText(
                img_np, f"{idx}, {self.CLASSES[label]}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (157, 80, 136), 2
            )

        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"test_debug_images/image_{self.cur_iter}.png", img_np)

    def generate_pseudo_labels(self, teacher_outputs, img_meta, **kwargs):
        """Generate pseudo label for UnbiasedTeacher."""
        device = kwargs.pop("device")
        all_pseudo_bboxes = []
        all_pseudo_labels = []
        all_pseudo_masks = []
        num_all_bboxes = 0
        num_all_pseudo = 0
        for i, teacher_bboxes_labels in enumerate(teacher_outputs):
            image_shape = img_meta[i]["img_shape"][:-1]
            pseudo_bboxes = []
            pseudo_labels = []
            pseudo_masks = []
            if self.model_t.with_mask:
                teacher_bboxes_labels = zip(*teacher_bboxes_labels)
            for label, teacher_bboxes_masks in enumerate(teacher_bboxes_labels):
                if self.model_t.with_mask:
                    teacher_bboxes = teacher_bboxes_masks[0]
                    teacher_masks = teacher_bboxes_masks[1]
                else:
                    teacher_bboxes = teacher_bboxes_masks
                confidences = teacher_bboxes[:, -1]
                pseudo_indices = confidences > self.pseudo_conf_thresh
                pseudo_bboxes.append(teacher_bboxes[pseudo_indices, :4])  # model output: [x y w h conf]
                pseudo_labels.append(np.full([sum(pseudo_indices)], label))
                if self.model_t.with_mask:
                    if np.any(pseudo_indices):
                        teacher_masks = [np.expand_dims(mask, 0) for mask in teacher_masks]
                        pseudo_masks.append(np.concatenate(teacher_masks)[pseudo_indices])
                    else:
                        pseudo_masks.append(np.array([]).reshape(0, *image_shape))

                num_all_bboxes += teacher_bboxes.shape[0]
                if len(pseudo_bboxes):
                    num_all_pseudo += pseudo_bboxes[-1].shape[0]

            if len(pseudo_bboxes) > 0:
                all_pseudo_bboxes.append(torch.from_numpy(np.concatenate(pseudo_bboxes)).to(device))
                all_pseudo_labels.append(torch.from_numpy(np.concatenate(pseudo_labels)).to(device))
                if self.model_t.with_mask:
                    all_pseudo_masks.append(BitmapMasks(np.concatenate(pseudo_masks), *image_shape))

        pseudo_ratio = float(num_all_pseudo) / num_all_bboxes if num_all_bboxes > 0 else 0.0
        return all_pseudo_bboxes, all_pseudo_labels, all_pseudo_masks, pseudo_ratio

    @staticmethod
    def _update_unlabeled_loss(sum_loss, loss, loss_name, weight):
        if isinstance(loss, list):
            sum_loss[loss_name + "_ul"] = [cur_loss * weight for cur_loss in loss]
        else:
            sum_loss[loss_name + "_ul"] = loss * weight

    @staticmethod
    def state_dict_hook(module, state_dict, prefix, *args, **kwargs):  # pylint: disable=unused-argument
        """Redirect student model as output state_dict (teacher as auxilliary)."""
        logger.info("----------------- MeanTeacherSegmentor.state_dict_hook() called")
        for key in list(state_dict.keys()):
            value = state_dict.pop(key)
            if not prefix or key.startswith(prefix):
                key = key.replace(prefix, "", 1)
                if key.startswith("model_t."):
                    key = key.replace("model_t.", "", 1)
                elif key.startswith("model_s."):
                    continue
                key = prefix + key
            state_dict[key] = value
        return state_dict

    @staticmethod
    def load_state_dict_pre_hook(module, state_dict, *args, **kwargs):  # pylint: disable=unused-argument
        """Redirect input state_dict to teacher model."""
        logger.info("----------------- MeanTeacherSegmentor.load_state_dict_pre_hook() called")
        for key in list(state_dict.keys()):
            value = state_dict.pop(key)
            state_dict["model_s." + key] = value
            state_dict["model_t." + key] = value
