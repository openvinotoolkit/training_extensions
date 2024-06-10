# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

"""Custom 3D recognizer for OTX."""
from __future__ import annotations

from typing import Any

import torch

from otx.algo.action_classification.utils.data_sample import ActionDataSample
from otx.algo.modules.base_module import BaseModule


class BaseRecognizer(BaseModule):
    """Custom 3d recognizer class for OTX.

    This is for patching forward function during export procedure.
    """

    def __init__(
        self,
        backbone: torch.Module,
        cls_head: torch.Module,
        neck: torch.Module | None = None,
        test_cfg: dict | None = None,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.cls_head = cls_head
        if neck is not None:
            self.neck = neck
        self.test_cfg = test_cfg

    @property
    def with_neck(self) -> bool:
        """bool: whether the recognizer has a neck."""
        return hasattr(self, "neck") and self.neck is not None

    @property
    def with_cls_head(self) -> bool:
        """bool: whether the recognizer has a cls_head."""
        return hasattr(self, "cls_head") and self.cls_head is not None

    def extract_feat(
        self,
        inputs: torch.Tensor,
        stage: str = "neck",
        data_samples: list[ActionDataSample] | None = None,
        test_mode: bool = False,
    ) -> tuple:
        """Extract features of different stages.

        Args:
            inputs (torch.Tensor): The input data.
            stage (str): Which stage to output the feature.
                Defaults to ``'neck'``.
            data_samples (list[:obj:`ActionDataSample`], optional): Action data
                samples, which are only needed in training. Defaults to None.
            test_mode (bool): Whether in test mode. Defaults to False.

        Returns:
                torch.Tensor: The extracted features.
                dict: A dict recording the kwargs for downstream
                    pipeline. These keys are usually included:
                    ``loss_aux``.
        """
        # Record the kwargs required by `loss` and `predict`
        loss_predict_kwargs = {}

        num_segs = inputs.shape[1]
        # [N, num_crops, C, T, H, W] ->
        # [N * num_crops, C, T, H, W]
        # `num_crops` is calculated by:
        #   1) `twice_sample` in `SampleFrames`
        #   2) `num_sample_positions` in `DenseSampleFrames`
        #   3) `ThreeCrop/TenCrop` in `test_pipeline`
        #   4) `num_clips` in `SampleFrames` or its subclass if `clip_len != 1`
        inputs = inputs.view((-1,) + inputs.shape[2:])

        # Check settings of test
        if test_mode:
            if self.test_cfg is not None:
                loss_predict_kwargs["fcn_test"] = self.test_cfg.get("fcn_test", False)
            if self.test_cfg is not None and self.test_cfg.get("max_testing_views", False):
                max_testing_views = self.test_cfg.get("max_testing_views")
                if not isinstance(max_testing_views, int):
                    msg = "max_testing_views should be 'int'"
                    raise TypeError(msg)

                total_views = inputs.shape[0]
                if num_segs != total_views:
                    msg = "max_testing_views is only compatible with batch_size == 1"
                    raise ValueError(msg)
                view_ptr = 0
                feats = []
                while view_ptr < total_views:
                    batch_imgs = inputs[view_ptr : view_ptr + max_testing_views]
                    feat = self.backbone(batch_imgs)
                    if self.with_neck:
                        feat, _ = self.neck(feat)
                    feats.append(feat)
                    view_ptr += max_testing_views

                def recursively_cat(
                    feats: torch.Tensor | list[Any] | tuple[Any, ...],
                ) -> tuple[torch.Tensor, ...]:
                    # recursively traverse feats until it's a tensor,
                    # then concat
                    out_feats: list[torch.Tensor] = []
                    for e_idx, elem in enumerate(feats[0]):
                        batch_elem = [feat[e_idx] for feat in feats]
                        if not isinstance(elem, torch.Tensor):
                            batch_elem = recursively_cat(batch_elem)  # type: ignore[assignment]
                        else:
                            batch_elem = torch.cat(batch_elem)
                        out_feats.append(batch_elem)

                    return tuple(out_feats)

                x = recursively_cat(feats) if isinstance(feats[0], tuple) else torch.cat(feats)
            else:
                x = self.backbone(inputs)
                if self.with_neck:
                    x, _ = self.neck(x)

            return x, loss_predict_kwargs

        # Return features extracted through backbone
        x = self.backbone(inputs)
        if stage == "backbone":
            return x, loss_predict_kwargs

        loss_aux = {}
        if self.with_neck:
            x, loss_aux = self.neck(x, data_samples=data_samples)

        # Return features extracted through neck
        loss_predict_kwargs["loss_aux"] = loss_aux
        if stage == "neck":
            return x, loss_predict_kwargs

        # Return raw logits through head.
        x = self.cls_head(x, **loss_predict_kwargs)
        return x, loss_predict_kwargs

    def forward(
        self,
        inputs: torch.Tensor,
        data_samples: list[ActionDataSample] | None = None,
        mode: str = "tensor",
        **kwargs,
    ) -> dict[str, torch.Tensor] | list[ActionDataSample] | tuple[torch.Tensor] | torch.Tensor:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes:

        - ``tensor``: Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - ``predict``: Forward and return the predictions, which are fully
        processed to a list of :obj:`ActionDataSample`.
        - ``loss``: Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[``ActionDataSample], optional): The
                annotation data of every samples. Defaults to None.
            mode (str): Return what kind of value. Defaults to ``tensor``.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of ``ActionDataSample``.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == "predict":
            return self.predict(inputs, data_samples, **kwargs)
        if mode == "loss":
            return self.loss(inputs, data_samples, **kwargs)
        if mode == "tensor":
            return self._forward(inputs, **kwargs)

        msg = f"Invalid mode '{mode}'. Only supports loss, predict and tensor mode"
        raise RuntimeError(msg)

    def loss(self, inputs: torch.Tensor, data_samples: list[ActionDataSample] | None, **kwargs) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (torch.Tensor): Raw Inputs of the recognizer.
                These should usually be mean centered and std scaled.
            data_samples (List[``ActionDataSample``]): The batch
                data samples. It usually includes information such
                as ``gt_label``.

        Returns:
            dict: A dictionary of loss components.
        """
        feats, loss_kwargs = self.extract_feat(inputs, data_samples=data_samples)

        # loss_aux will be a empty dict if `self.with_neck` is False.
        loss_aux = loss_kwargs.get("loss_aux", {})
        loss_cls = self.cls_head.loss(feats, data_samples, **loss_kwargs)
        return self._merge_dict(loss_cls, loss_aux)

    def predict(
        self,
        inputs: torch.Tensor,
        data_samples: list[ActionDataSample] | None,
        **kwargs,
    ) -> list[ActionDataSample]:
        """Predict results from a batch of inputs and data samples with postprocessing.

        Args:
            inputs (torch.Tensor): Raw Inputs of the recognizer.
                These should usually be mean centered and std scaled.
            data_samples (List[``ActionDataSample``]): The batch
                data samples. It usually includes information such
                as ``gt_label``.

        Returns:
            List[``ActionDataSample``]: Return the recognition results.
            The returns value is ``ActionDataSample``, which usually contains
            ``pred_scores``. And the ``pred_scores`` usually contains
            following keys.

                - item (torch.Tensor): Classification scores, has a shape
                    (num_classes, )
        """
        feats, predict_kwargs = self.extract_feat(inputs, test_mode=True)
        return self.cls_head.predict(feats, data_samples, **predict_kwargs)

    def _forward(self, inputs: torch.Tensor, stage: str = "backbone", **kwargs) -> torch.Tensor:
        """Network forward process for export procedure.

        Args:
            inputs (torch.Tensor): Raw Inputs of the recognizer.
            stage (str): Which stage to output the features.

        """
        feats, predict_kwargs = self.extract_feat(inputs, test_mode=True)
        cls_scores = self.cls_head(feats, **predict_kwargs)
        num_segs = cls_scores.shape[0] // inputs.shape[1]
        return self.cls_head.average_clip(cls_scores, num_segs=num_segs)

    @staticmethod
    def _merge_dict(*args) -> dict:
        """Merge all dictionaries into one dictionary.

        If pytorch version >= 1.8, ``merge_dict`` will be wrapped
        by ``torch.fx.wrap``,  which will make ``torch.fx.symbolic_trace`` skip
        trace ``merge_dict``.

        Note:
            If a function needs to be traced by ``torch.fx.symbolic_trace``,
            but inevitably needs to use ``update`` method of ``dict``(``update``
            is not traceable). It should use ``merge_dict`` to replace
            ``xxx.update``.

        Args:
            *args: dictionary needs to be merged.

        Returns:
            dict: Merged dict from args
        """
        output = {}
        for item in args:
            if not isinstance(item, dict):
                msg = f"all arguments of merge_dict should be a dict, but got {type(item)}"
                raise TypeError(msg)
            output.update(item)
        return output
