"""UnbiasedTeacher Class for mmdetection detectors."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import copy
import functools
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import torch
import mmcv
import cv2
from mmdet.core import bbox2result, bbox2roi
from mmdet.core.mask.structures import BitmapMasks
from mmdet.models import DETECTORS, build_detector
from mmdet.models.detectors import BaseDetector

from otx.algorithms.common.utils.logger import get_logger

from .sam_detector_mixin import SAMDetectorMixin

logger = get_logger()

# TODO: Need to fix pylint issues
# pylint: disable=abstract-method, too-many-locals, unused-argument


@DETECTORS.register_module()
class UnbiasedTeacher(SAMDetectorMixin, BaseDetector):
    """Mean teacher framework for detection and instance segmentation."""

    def __init__(
        self,
        arch_type,
        unlabeled_loss_weights={"cls": 1.0, "bbox": 1.0, "mask": 1.0},
        pseudo_conf_thresh=0.7,
        bg_loss_weight=-1.0,
        min_pseudo_label_ratio=0.0,
        two_stage=False,
        **kwargs
    ):
        super().__init__()
        self.unlabeled_loss_weights = unlabeled_loss_weights
        self.pseudo_conf_thresh = pseudo_conf_thresh
        self.bg_loss_weight = bg_loss_weight
        self.min_pseudo_label_ratio = min_pseudo_label_ratio
        cfg = kwargs.copy()
        cfg["type"] = arch_type
        self.debug = True
        self.two_stage = two_stage
        self.model_s = build_detector(cfg)
        self.model_t = build_detector(cfg)
        self.model_t.eval()
        self.iter = 0
        # warmup for first epochs
        self.enable_unlabeled_loss(False)

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

    @torch.no_grad()
    def inference_unlabeled(self, img, img_metas, rescale=True, return_feat=False):
        # inference: create pseudo label
        x = self.model_t.extract_feat(img)
        if self.two_stage:
            proposal_list = self.model_t.rpn_head.simple_test_rpn(x, img_metas)
            # bboxes
            bbox_results = self.model_t.roi_head.simple_test_bboxes(
                x, img_metas, proposal_list, self.model_t.roi_head.test_cfg, rescale=rescale)
        else:
            outs = self.model_t.bbox_head(x)
            # det_bboxes, det_labels = self.model_t.bbox_head.get_bboxes(*outs, img_metas=img_metas, cfg=self.model_t.test_cfg)
            bbox_results = self.model_t.bbox_head.simple_test(x, img_metas, rescale=True)
        breakpoint()
        return bbox_results

    def create_pseudo_results(self, img, bbox_results, box_transform, device,
                              gt_bboxes=None, gt_labels=None, img_metas=None):
        """using dynamic score to create pseudo results"""
        gt_bboxes_pred, gt_labels_pred = [], []
        _, _, h, w = img.shape
        use_gt = gt_bboxes is not None
        for b, result in enumerate(bbox_results):
            bboxes, labels = [], []
            for cls, bb in enumerate(result):
                label = cls * np.ones_like(bb[:, 0], dtype=np.uint8)
                flag = bb[:, -1] >= self.pseudo_conf_thresh
                bboxes.append(bb[flag][:, :4])
                labels.append(label[flag])
            bboxes = np.concatenate(bboxes)
            labels = np.concatenate(labels)
            # for bf in box_transform[b]:
            #     bboxes, labels = bf(bboxes, labels)
            gt_bboxes_pred.append(torch.from_numpy(bboxes).float().to(device))
            gt_labels_pred.append(torch.from_numpy(labels).long().to(device))
        return gt_bboxes_pred, gt_labels_pred

    def forward_train(self, img, img_metas, img0, gt_bboxes, gt_labels, gt_bboxes_ignore=None, **kwargs):
        if self.iter < 10000:
            record_dict = self.model_s.forward_train(
                torch.cat((img0, img)),  # weak + hard augmented images
                img_metas + img_metas,
                gt_bboxes + gt_bboxes,
                gt_labels + gt_labels,
                gt_bboxes_ignore + gt_bboxes_ignore if gt_bboxes_ignore else None,
            )
            self._update_teacher_model(keep_rate=0.00)
            self.iter += 1
            return record_dict

        else:
            if self.iter == 10000:
                print("!!!!!!!!!!!!!ACHTUNG!!!!!!!!!!!!!!")
                self._update_teacher_model(keep_rate=0.00)

        self.iter += 1

        ul_args = kwargs.get("extra_0", {})
        ul_img = ul_args.get("img")
        ul_img0 = ul_args.get("img0") # CHECK that this is not hard augmented
        ul_img_metas = ul_args.get("img_metas") # Why meta is only one? Need be 2

        with torch.no_grad():
            # WE NEED RAW OUTPUTS + NMS
            teacher_results = self.model_t.forward_test([ul_img0], [ul_img_metas], rescale=True)  # easy augmentation
            # teacher_results = self.inference_unlabeled(ul_img0, ul_img_metas)
            bbox_transform = []
            # for img_meta in ul_img_metas:
            #     bbox_transform.append(img_meta.pop('bbox_transform'))

            gt_bboxes_pred, gt_labels_pred = self.create_pseudo_results(img, bbox_results=teacher_results,
                                                                        box_transform=bbox_transform, device=ul_img0.device)

            if self.debug:
                self.visual_online(ul_img, gt_bboxes_pred, gt_labels_pred)

        # use the above raw teacher prediction and perform another NMS (NMS_CRITERIA_REG_TRAIN)
        # FIGURE OUT - we need one more NMS
        losses_unlabeled = self.model_s.forward_train(ul_img, ul_img_metas,
                                            gt_bboxes_pred, gt_labels_pred)
        # process loss
        # ADD IT
        # supervised loss
        record_dict = self.model_s.forward_train(
            torch.cat((img0, img)),  # weak + hard augmented images
            img_metas + img_metas,
            gt_bboxes + gt_bboxes,
            gt_labels + gt_labels,
            gt_bboxes_ignore + gt_bboxes_ignore if gt_bboxes_ignore else None,
        )

        # breakpoint()
        for key, val in losses_unlabeled.items():
            if key.find('loss') == -1:
                continue
            if key.find('bbox') != -1:
                # record_dict[key + "_ul"] = self.unlabeled_loss_weights["bbox"] * val
                record_dict[key + "_ul"] = val
            elif key.find('cls') != -1:
                # record_dict[key+ "_ul"] = self.unlabeled_loss_weights["cls"] * val
                record_dict[key+ "_ul"] =  val
            else:
                # record_dict[key+ "_ul"] = self.unlabeled_loss_weights["centerness"] * val
                record_dict[key+ "_ul"] = val

        # breakpoint()
        self._update_teacher_model()
        return record_dict

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.996):
        student_model_dict = self.model_s.state_dict()
        teacher_model_dict = self.model_t.state_dict()
        new_teacher_dict = OrderedDict()

        for key, value in teacher_model_dict.items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] * (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        for key, value in new_teacher_dict.items():
            teacher_model_dict[key].copy_(value)

    def visual_online(self, img, boxes_list, labels_list, img_id=0,
                      boxes_ignore_list=None, proposal_list=None):
        img_norm_cfg = dict(
            mean=np.array([123.675, 116.28, 103.53]), std=np.array([58.395, 57.12, 57.375])
        )
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
                cv2.putText(img_np, f'{idx}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (214, 39, 40), 2)
        # ignore
        if boxes_ignore_list:
            boxes_ignore = boxes_ignore_list[img_id]
            for idx, box in enumerate(boxes_ignore):
                x1, y1, x2, y2 = [int(a.cpu().item()) for a in box]
                img_np = cv2.rectangle(img_np, (x1, y1), (x2, y2), (44, 160, 44), 2)
                cv2.putText(img_np, f'{idx}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (44, 160, 44), 2)
        # pseudo gt
        for idx, (box, label) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = [int(a.cpu().item()) for a in box]
            img_np = cv2.rectangle(img_np, (x1, y1), (x2, y2), (157, 80, 136), 2)
            cv2.putText(img_np, f'{idx}, {self.CLASSES[label]}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (157, 80, 136), 2)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"image_{img_id}.png", img_np)
        plt.show()

    @staticmethod
    def state_dict_hook(module, state_dict, prefix, *args, **kwargs):  # pylint: disable=unused-argument
        """Redirect student model as output state_dict (teacher as auxilliary)."""
        logger.info("----------------- MeanTeacherSegmentor.state_dict_hook() called")
        for key in list(state_dict.keys()):
            value = state_dict.pop(key)
            if not prefix or key.startswith(prefix):
                key = key.replace(prefix, "", 1)
                if key.startswith("model_s."):
                    key = key.replace("model_s.", "", 1)
                elif key.startswith("model_t."):
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
