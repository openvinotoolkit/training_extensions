"""Mean teacher segmentor for semi-supervised learning."""

import functools

import numpy as np
import torch
from mmseg.models import SEGMENTORS, build_segmentor
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.ops import resize

from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.segmentation.adapters.mmseg.models.heads.proto_head import ProtoNet

logger = get_logger()

# pylint: disable=too-many-locals, protected-access


@SEGMENTORS.register_module()
class MeanTeacherSegmentor(BaseSegmentor):
    """Mean teacher segmentor for semi-supervised learning.

    It creates two models and ema from one to the other for consistency loss.

    Args:
        orig_type (BaseSegmentor): original type of segmentor to build student and teacher models
        num_iters_per_epoch (int): number of iterations per training epoch.
        unsup_weight (float): loss weight for unsupervised part. Default: 0.1
        proto_weight (float): loss weight for pixel prototype cross entropy loss. Default: 0.7
        drop_unrel_pixels_percent (int): starting precentage of pixels with high entropy
            to drop from teachers pseudo labels. Default: 20
        semisl_start_epoch (int): epoch to start learning with unlabeled images. Default: 1
        proto_head (dict): configuration to constract prototype network. Default: None
    """

    def __init__(
        self,
        orig_type,
        num_iters_per_epoch=None,
        unsup_weight=0.1,
        proto_weight=0.7,
        drop_unrel_pixels_percent=20,
        semisl_start_epoch=1,
        proto_head=None,
        **kwargs
    ):
        super().__init__()
        self.test_cfg = kwargs["test_cfg"]
        self.count_iter = 0
        # num_iters_per_epoch will be None during validation
        # Overwise it should be overwritten in train_task
        if num_iters_per_epoch is not None:
            # filter unreliable pixels during first 100 epochs
            self.filter_pixels_iters = num_iters_per_epoch * 100
            self.semisl_start_iter = num_iters_per_epoch * semisl_start_epoch
        self.drop_unrel_pixels_percent = drop_unrel_pixels_percent
        cfg = kwargs.copy()
        cfg["type"] = orig_type
        self.align_corners = cfg["decode_head"].get("align_corners", False)
        self.model_s = build_segmentor(cfg)
        self.model_t = build_segmentor(cfg)
        self.use_prototype_head = False
        self.unsup_weight = unsup_weight
        if proto_head is not None and hasattr(self.model_s.decode_head, "_forward_feature"):
            self.proto_net = ProtoNet(num_classes=self.model_s.decode_head.num_classes, **proto_head)
            self.use_prototype_head = True
        elif proto_head is not None:
            logger.warning(
                "Prototype head isn't supported by this model. "
                "_forward_feature() method is required to be presented in the main decode head. "
                "This function will be disabled and standard Mean Teacher algorithm will be utilized"
            )
        self.proto_weight = proto_weight
        self.losses = dict()
        # Hooks for super_type transparent weight load/save
        self._register_state_dict_hook(self.state_dict_hook)
        self._register_load_state_dict_pre_hook(functools.partial(self.load_state_dict_pre_hook, self))

    def encode_decode(self, img, img_metas):
        """Encode and decode images."""
        return self.model_s.encode_decode(img, img_metas)

    def decode_proto_network(
        self, sup_input, gt_semantic_seg, unsup_input=None, pl_from_teacher=None, reweight_unsup=1.0
    ):
        """Forward prototype network, compute proto loss.

        If there is no unsupervised part, only supervised loss will be computed.

        Args:
            sup_input (torch.Tensor): student output from labeled images
            gt_semantic_seg (torch.Tensor): ground truth semantic segmentation label maps
            unsup_input (torch.Tensor): student output from unlabeled images. Default: None
            pl_from_teacher (torch.Tensor): teacher generated pseudo labels. Default: None
            reweight_unsup (float): reweighting coefficient for unsupervised part after
                filtering high entropy pixels. Default: 1.0
        """

        # supervised branch
        head_features_sup = self.model_s.decode_head._forward_feature(sup_input)
        proto_out_supervised = self.proto_net(head_features_sup, gt_semantic_seg)
        loss_proto = self.proto_net.losses(**proto_out_supervised, seg_label=gt_semantic_seg)
        self._update_summary_loss(loss_proto, loss_weight=self.proto_weight)
        # unsupervised branch
        if unsup_input is not None and pl_from_teacher is not None:
            head_features_unsup = self.model_s.decode_head._forward_feature(unsup_input)
            proto_out_unsupervised = self.proto_net(head_features_unsup, pl_from_teacher)
            loss_proto_u = self.proto_net.losses(**proto_out_unsupervised, seg_label=pl_from_teacher)
            self._update_summary_loss(loss_proto_u, loss_weight=self.unsup_weight * reweight_unsup * self.proto_weight)

    def generate_pseudo_labels(self, ul_w_img, ul_img_metas):
        """Generate pseudo labels from teacher model, apply filter loss method.

        Args:
            ul_w_img (torch.Tensor): weakly augmented unlabeled images
            ul_img_metas (dict): unlabeled images meta data

        """

        with torch.no_grad():
            teacher_feat = self.model_t.extract_feat(ul_w_img)
            teacher_out = self.model_t._decode_head_forward_test(teacher_feat, ul_img_metas)
            teacher_out = resize(
                input=teacher_out, size=ul_w_img.shape[2:], mode="bilinear", align_corners=self.align_corners
            )
            teacher_prob_unsup = torch.softmax(teacher_out, axis=1)
            _, pl_from_teacher = torch.max(teacher_prob_unsup, axis=1, keepdim=True)

        # drop pixels with high entropy
        percent_unreliable = self.drop_unrel_pixels_percent * (1 - self.count_iter / self.filter_pixels_iters)
        reweight_unsup = 1.0
        if percent_unreliable > 0:
            keep_percent = 100 - percent_unreliable
            batch_size, _, h, w = teacher_out.shape

            entropy = -torch.sum(teacher_prob_unsup * torch.log(teacher_prob_unsup + 1e-10), dim=1, keepdim=True)

            thresh = np.percentile(entropy[pl_from_teacher != 255].detach().cpu().numpy().flatten(), keep_percent)
            thresh_mask = entropy.ge(thresh).bool() * (pl_from_teacher != 255).bool()

            pl_from_teacher[thresh_mask] = 255
            # reweight unsupervised loss
            reweight_unsup = batch_size * h * w / torch.sum(pl_from_teacher != 255)

        return pl_from_teacher, reweight_unsup

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
        """Forward train.

        Args:
            img (torch.Tensor): labeled images
            img_metas (dict): labeled images meta data
            gt_semantic_seg (torch.Tensor): semantic segmentation label maps
            kwargs (dict): key arguments with unlabeled components and additional information
        """
        self.count_iter += 1
        self.losses["sum_loss"] = 0.0
        if self.semisl_start_iter >= self.count_iter or "extra_0" not in kwargs:
            x = self.model_s.extract_feat(img)
            loss_decode = self.model_s._decode_head_forward_train(x, img_metas, gt_semantic_seg=gt_semantic_seg)
            self._update_summary_loss(loss_decode)
            if self.use_prototype_head:
                self.decode_proto_network(x, gt_semantic_seg)

            # add information about accuracy
            for key in loss_decode:
                if "acc" in key and loss_decode[key] is not None:
                    self.losses["decode_acc"] = loss_decode[key]

            return self.losses

        # + unsupervised part
        ul_data = kwargs["extra_0"]
        ul_s_img = ul_data["img"]  # strongly augmented
        ul_w_img = ul_data["ul_w_img"]  # weakly augmented
        ul_img_metas = ul_data["img_metas"]

        # generate pseudo labels, filter high entropy pixels, compute loss reweight
        pl_from_teacher, reweight_unsup = self.generate_pseudo_labels(ul_w_img, ul_img_metas)

        # extract features from labeled and unlabeled augmented images
        x = self.model_s.extract_feat(img)
        x_u = self.model_s.extract_feat(ul_s_img)
        loss_decode = self.model_s._decode_head_forward_train(x, img_metas, gt_semantic_seg=gt_semantic_seg)
        loss_decode_u = self.model_s._decode_head_forward_train(x_u, ul_img_metas, gt_semantic_seg=pl_from_teacher)
        self._update_summary_loss(loss_decode)
        self._update_summary_loss(loss_decode_u, loss_weight=self.unsup_weight * reweight_unsup)

        if self.use_prototype_head:
            # for proto head we need to derive head features
            self.decode_proto_network(x, gt_semantic_seg, x_u, pl_from_teacher, reweight_unsup)

        # add information about accuracy
        for key in loss_decode:
            if "acc" in key and loss_decode[key] is not None:
                self.losses["decode_acc"] = loss_decode[key]
                self.losses["decode_acc_unsup"] = loss_decode_u[key]

        return self.losses

    def _update_summary_loss(self, decode_loss, loss_weight=1.0):
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
