"""Mean teacher segmentor for semi-supervised learning."""

import functools

import torch
from mmseg.models import SEGMENTORS, build_segmentor
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.ops import resize
import numpy as np

from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.segmentation.adapters.mmseg.models.heads.proto_head import ProtoNet

logger = get_logger()

# pylint: disable=too-many-locals, protected-access


@SEGMENTORS.register_module()
class MeanTeacherSegmentor(BaseSegmentor):
    """Mean teacher segmentor for semi-supervised learning.

    It creates two models and ema from one to the other for consistency loss.
    """

    def __init__(self, orig_type=None,
                 unsup_weight=0.1,
                 proto_weight=0.7,
                 aux_weight=0.1,
                 drop_percent=80,
                 num_iters_per_epoch=6000,
                 semisl_start_iter=1,
                 proto_head=None,
                 **kwargs):
        super().__init__()
        self.test_cfg = kwargs["test_cfg"]
        self.count_iter = 0
        self.filter_pixels_iters = num_iters_per_epoch * 100 # 100 epochs
        self.semisl_start_iter = num_iters_per_epoch * semisl_start_iter # 1 epoch
        self.drop_percent = drop_percent
        self.aux_weight = aux_weight
        cfg = kwargs.copy()
        cfg["type"] = orig_type
        self.align_corners = cfg["decode_head"].align_corners
        self.model_s = build_segmentor(cfg)
        self.model_t = build_segmentor(cfg)
        self.use_prototype_head = False
        self.unsup_weight = unsup_weight
        if proto_head is not None and hasattr(self.model_s.decode_head, "_forward_feature"):
            self.proto_net = ProtoNet(num_classes=self.model_s.decode_head.num_classes, **proto_head)
            self.use_prototype_head = True
        elif proto_head is not None:
            logger.warning("Prototype head isn't supported by this model. "
                           "_forward_feature() method is required to be presented in the main decode head")
        self.proto_weight = proto_weight
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
        self.losses["sum_loss"] = 0.0
        if self.semisl_start_iter > self.count_iter or "extra_0" not in kwargs:
            x = self.model_s.extract_feat(img)
            if self.use_prototype_head:
                head_features = self.model_s.decode_head._forward_feature(x)
                out = self.model_s.decode_head._forward_cls(head_features)
                loss_decode = self.model_s.decode_head.forward_train(out, img_metas, gt_semantic_seg=gt_semantic_seg, loss_only=True)
                self.update_summary_loss(loss_decode)
                proto_out = self.proto_net(head_features, gt_semantic_seg, orig_size=img.shape[2:])
                loss_proto = self.proto_net.losses(**proto_out, seg_label=gt_semantic_seg)
                self.update_summary_loss(loss_proto, self.proto_weight)
            else:
                loss_decode = self.model_s._decode_head_forward_train(x, img_metas, gt_semantic_seg=gt_semantic_seg)
                self.update_summary_loss(loss_decode)

            self.losses["decode_acc"] = loss_decode["decode.acc_seg"]
            return self.losses

        # + unsupervised part
        ul_data = kwargs["extra_0"]
        ul_s_img = ul_data["img"] # strongly augmented
        ul_w_img = ul_data["ul_w_img"] # weakly augmented
        ul_img_metas = ul_data["img_metas"]

        with torch.no_grad():
            teacher_feat = self.model_t.extract_feat(ul_w_img)
            teacher_out = self.model_t._decode_head_forward_test(teacher_feat, ul_img_metas)
            teacher_out = resize(
                input=teacher_out, size=ul_w_img.shape[2:], mode="bilinear", align_corners=self.align_corners
            )
            teacher_prob_unsup = torch.softmax(teacher_out, axis=1)
            _, pl_from_teacher = torch.max(teacher_prob_unsup, axis=1, keepdim=True)

        # drop pixels with high entropy
        drop_percent = self.drop_percent
        percent_unreliable = (100 - drop_percent) * (1 - self.count_iter / self.filter_pixels_iters)
        drop_percent = 100 if percent_unreliable <= 0 else 100 - percent_unreliable
        batch_size, _, h, w = teacher_out.shape

        entropy = -torch.sum(teacher_prob_unsup * torch.log(teacher_prob_unsup + 1e-10), dim=1, keepdim=True)

        thresh = np.percentile(
            entropy[pl_from_teacher != 255].detach().cpu().numpy().flatten(), drop_percent
        )
        thresh_mask = entropy.ge(thresh).bool() * (pl_from_teacher != 255).bool()

        pl_from_teacher[thresh_mask] = 255
        reweight_unsup = batch_size * h * w / torch.sum(pl_from_teacher != 255)

        # extract features from labeled and unlabeled augmented images
        x = self.model_s.extract_feat(img)
        x_u = self.model_s.extract_feat(ul_s_img)
        if self.use_prototype_head:
            # for proto head we need to derive head features
            head_features_sup = self.model_s.decode_head._forward_feature(x)
            head_features_unsup = self.model_s.decode_head._forward_feature(x_u)
            out_sup = self.model_s.decode_head._forward_cls(head_features_sup)
            out_unsup = self.model_s.decode_head._forward_cls(head_features_unsup)
            # proto forward + proto learning
            proto_out_supervised = self.proto_net(head_features_sup, gt_semantic_seg, orig_size=img.shape[2:])
            proto_out_unsupervised = self.proto_net(head_features_unsup, pl_from_teacher, orig_size=img.shape[2:])
            # compute losses
            loss_decode = self.model_s.decode_head.forward_train(out_sup, img_metas, gt_semantic_seg=gt_semantic_seg, loss_only=True)
            loss_decode_u = self.model_s.decode_head.forward_train(out_unsup, ul_img_metas, gt_semantic_seg=pl_from_teacher, loss_only=True)
            self.update_summary_loss(loss_decode)
            self.update_summary_loss(loss_decode_u, loss_weight=self.unsup_weight * reweight_unsup)
            # proto loss computation
            loss_proto = self.proto_net.losses(**proto_out_supervised, seg_label=gt_semantic_seg)
            loss_proto_u = self.proto_net.losses(**proto_out_unsupervised, seg_label=pl_from_teacher)
            self.update_summary_loss(loss_proto, oss_weight=self.proto_weight)
            self.update_summary_loss(loss_proto_u, loss_weight=self.unsup_weight * reweight_unsup * self.proto_weight)
        else:
            loss_decode = self.model_s._decode_head_forward_train(out_sup, img_metas, gt_semantic_seg=gt_semantic_seg)
            loss_decode_u = self.model_s._decode_head_forward_train(out_unsup, ul_img_metas, gt_semantic_seg=pl_from_teacher)
            self.update_summary_loss(loss_decode)
            self.update_summary_loss(loss_decode_u, loss_weight=self.unsup_weight * reweight_unsup)


        if hasattr(self.model_s, "auxiliary_head"):
            aux_loss = self.model_s.auxiliary_head.forward_train(x, img_metas, gt_semantic_seg=gt_semantic_seg)
            aux_loss_u = self.model_s.auxiliary_head.forward_train(x_u, ul_img_metas, gt_semantic_seg=pl_from_teacher)
            self.update_summary_loss(aux_loss, loss_weight=self.aux_weight)
            self.update_summary_loss(aux_loss_u, loss_weight=self.aux_weight * reweight_unsup * self.unsup_weight)

        # add information about accuracy
        self.losses["decode_acc"] = loss_decode["decode.acc_seg"]
        self.losses["decode_acc_unsup"] = loss_decode_u["decode.acc_seg"]

        return self.losses

    def update_summary_loss(self, decode_loss, loss_weight=1.):
        for name, value in decode_loss.items():
            if value is None or "loss" not in name:
                continue
            self.losses["sum_loss"] += value * loss_weight

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
