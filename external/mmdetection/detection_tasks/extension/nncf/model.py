# Copyright (C) 2018-2021 OpenMMLab
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import cv2
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
import torch.distributed as dist
import torch.nn as nn
import warnings
from contextlib import contextmanager
from functools import partial
from mmcv.runner import auto_fp16

from .utils import no_nncf_trace, is_nncf_enabled
from mmdet.utils import get_root_logger
from mmdet.core import bbox2result


class NNCFDetectorMixin(nn.Module):
    """Base class for NNCF-enabled detectors."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fp16_enabled = False
        self.img_metas = None
        self.forward_backup = None

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.

        Note that if `kwargs` contains either `forward_export=True` or
        `dummy_forward=True` parameters, one of special branches of code is
        enabled for ONNX export
        (see the methods `forward_export` and `forward_dummy`).
        """
        if kwargs.get('forward_export'):
            return self.forward_export(imgs)

        if kwargs.get('dummy_forward'):
            return self.forward_dummy(imgs[0])

        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            with no_nncf_trace():
                return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.

        The parameter `img_metas` has the default value `None` for ONNX export only,
        in this case `return_loss` should be `False`, and `kwargs` should contain
        either `forward_dummy=True` or `dummy_forward=True` parameters to enable
        a special branch of code for ONNX tracer
        (see the method `forward_test`).
        """
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def forward_export(self, imgs):
        from torch.onnx.operators import shape_as_tensor
        assert self.img_metas, 'Error: forward_export should be called inside forward_export_context'

        img_shape = shape_as_tensor(imgs[0])
        imgs_per_gpu = int(imgs[0].size(0))
        assert imgs_per_gpu == 1
        assert len(self.img_metas[0]) == imgs_per_gpu, f'self.img_metas={self.img_metas}'
        self.img_metas[0][0]['img_shape'] = img_shape[2:4]

        return self.simple_test(imgs[0], self.img_metas[0], postprocess=False)
        # return self.simple_test(imgs[0], self.img_metas[0])

    @contextmanager
    def forward_export_context(self, img_metas):
        assert self.img_metas is None and self.forward_backup is None, 'Error: one forward context inside another forward context'

        if is_nncf_enabled():
            from nncf.torch.nncf_network import NNCFNetwork
            if isinstance(self, NNCFNetwork):
                self.get_nncf_wrapped_model().img_metas = img_metas
                self.get_nncf_wrapped_model().forward_backup = self.forward
        self.img_metas = img_metas
        self.forward_backup = self.forward
        self.forward = partial(self.forward, return_loss=False, forward_export=True, img_metas=None)
        yield
        self.forward = self.forward_backup
        self.forward_backup = None
        self.img_metas = None
        if is_nncf_enabled() and isinstance(self, NNCFNetwork):
            self.get_nncf_wrapped_model().img_metas = None
            self.get_nncf_wrapped_model().forward_backup = None

    @contextmanager
    def forward_dummy_context(self, img_metas):
        assert self.img_metas is None and self.forward_backup is None, 'Error: one forward context inside another forward context'

        self.img_metas = img_metas
        self.forward_backup = self.forward
        self.forward = partial(self.forward, return_loss=False, dummy_forward=True, img_metas=None)
        yield
        self.forward = self.forward_backup
        self.forward_backup = None
        self.img_metas = None

    def train_step(self, data, optimizer, compression_ctrl=None):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        if compression_ctrl is not None:
            compression_loss = compression_ctrl.loss()
            loss += compression_loss

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def export(self, img, img_metas, **kwargs):
        with self.forward_export_context(img_metas):
            torch.onnx.export(self, img, **kwargs)

    def simple_test(self, img, img_metas, rescale=False, postprocess=True):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        with no_nncf_trace():
            bbox_results = \
                self.bbox_head.get_bboxes(*outs, img_metas, self.test_cfg, False)
            if torch.onnx.is_in_onnx_export():
                feature_vector = get_feature_vector(x)
                saliency_map = get_saliency_map(x[-1])
                feature = feature_vector, saliency_map
                return bbox_results[0], feature

        if postprocess:
            bbox_results = [
                self.postprocess(det_bboxes, det_labels, None, img_metas, rescale=rescale)
                for det_bboxes, det_labels in bbox_results
            ]
        return bbox_results

    def postprocess(self,
                    det_bboxes,
                    det_labels,
                    det_masks,
                    img_meta,
                    rescale=False):
        num_classes = self.bbox_head.num_classes

        if rescale:
            scale_factor = img_meta[0]['scale_factor']
            if isinstance(det_bboxes, torch.Tensor):
                det_bboxes[:, :4] /= det_bboxes.new_tensor(scale_factor)
            else:
                det_bboxes[:, :4] /= np.asarray(scale_factor)

        bbox_results = bbox2result(det_bboxes, det_labels, num_classes)
        return bbox_results


class NNCFDenseHeadMixin(nn.Module):
    """Base class for NNCF-enabled DenseHeads."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        with no_nncf_trace():
            losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list
