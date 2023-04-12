"""Mean teacher segmentor for semi-supervised learning."""

import functools

import torch
from mmseg.models import SEGMENTORS, build_segmentor
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.ops import resize

from otx.algorithms.common.utils.logger import get_logger

logger = get_logger()

# pylint: disable=too-many-locals, protected-access


@SEGMENTORS.register_module()
class MeanTeacherSegmentor(BaseSegmentor):
    """Mean teacher segmentor for semi-supervised learning.

    It creates two models and ema from one to the other for consistency loss.
    """

    def __init__(self, orig_type=None, unsup_weight=0.1, semisl_start_iter=30, **kwargs):
        super().__init__()
        self.test_cfg = kwargs["test_cfg"]
        self.semisl_start_iter = semisl_start_iter
        self.count_iter = 0

        cfg = kwargs.copy()
        cfg["type"] = orig_type
        self.align_corners = cfg["decode_head"].align_corners
        self.model_s = build_segmentor(cfg)
        self.model_t = build_segmentor(cfg)
        self.unsup_weight = unsup_weight

        # Hooks for super_type transparent weight load/save
        self._register_state_dict_hook(self.state_dict_hook)
        self._register_load_state_dict_pre_hook(functools.partial(self.load_state_dict_pre_hook, self))

    def encode_decode(self, img, img_metas):
        """Encode and decode images."""
        return self.model_s.encode_decode(img, img_metas)

    def extract_feat(self, imgs):
        """Extract feature."""
        return self.model_s.extract_feat(imgs)

    def simple_test(self, img, img_meta, **kwargs):
        """Simple test."""
        return self.model_s.simple_test(img, img_meta, **kwargs)

    def aug_test(self, imgs, img_metas, **kwargs):
        """Aug test."""
        return self.model_s.aug_test(imgs, img_metas, **kwargs)

    def forward_dummy(self, img, **kwargs):
        """Forward dummy."""
        return self.model_s.forward_dummy(img, **kwargs)

    def forward_train(self, img, img_metas, gt_semantic_seg, **kwargs):
        """Forward train."""
        self.count_iter += 1
        if self.semisl_start_iter > self.count_iter or "extra_0" not in kwargs:
            x = self.model_s.extract_feat(img)
            loss_decode = self.model_s._decode_head_forward_train(x, img_metas, gt_semantic_seg=gt_semantic_seg)
            return loss_decode

        ul_data = kwargs["extra_0"]
        ul_s_img = ul_data["img"]
        ul_w_img = ul_data["ul_w_img"]
        ul_img_metas = ul_data["img_metas"]

        with torch.no_grad():
            teacher_feat = self.model_t.extract_feat(ul_w_img)
            teacher_logit = self.model_t._decode_head_forward_test(teacher_feat, ul_img_metas)
            teacher_logit = resize(
                input=teacher_logit, size=ul_w_img.shape[2:], mode="bilinear", align_corners=self.align_corners
            )
            _, pl_from_teacher = torch.max(torch.softmax(teacher_logit, axis=1), axis=1, keepdim=True)

        losses = dict()

        x = self.model_s.extract_feat(img)
        x_u = self.model_s.extract_feat(ul_s_img)
        loss_decode = self.model_s._decode_head_forward_train(x, img_metas, gt_semantic_seg=gt_semantic_seg)
        loss_decode_u = self.model_s._decode_head_forward_train(x_u, ul_img_metas, gt_semantic_seg=pl_from_teacher)

        for (key, value) in loss_decode_u.items():
            if value is None:
                continue
            losses[key] = loss_decode[key] + loss_decode_u[key] * self.unsup_weight

        return losses

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
