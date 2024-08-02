# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation of TopdownPoseEstimator for keypoint detection."""
from __future__ import annotations

from typing import TYPE_CHECKING

from otx.algo.modules.base_module import BaseModule

if TYPE_CHECKING:
    import torch
    from otx.algo.utils.mmengine_utils import InstanceData
    from otx.core.data.entity.keypoint_detection import KeypointDetBatchDataEntity
    from torch import Tensor, nn


class TopdownPoseEstimator(BaseModule):
    """Base class for top-down pose estimators.

    Args:
        backbone (dict): The backbone config
        neck (dict, optional): The neck config. Defaults to ``None``
        head (dict, optional): The head config. Defaults to ``None``
        train_cfg (dict, optional): The runtime config for training process.
            Defaults to ``None``
        test_cfg (dict, optional): The runtime config for testing process.
            Defaults to ``None``
        init_cfg (dict, optional): The config to control the initialization.
            Defaults to ``None``
        metainfo (dict): Meta information for dataset, such as keypoints
            definition and properties. If set, the metainfo of the input data
            batch will be overridden. For more details, please refer to
            https://mmpose.readthedocs.io/en/latest/user_guides/
            prepare_datasets.html#create-a-custom-dataset-info-
            config-file-for-the-dataset. Defaults to ``None``
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        neck: nn.Module | None = None,
        train_cfg: dict | None = None,
        test_cfg: dict | None = None,
        init_cfg: dict | list[dict] | None = None,
    ) -> None:
        super().__init__(init_cfg)

        self.backbone = backbone
        self.head = head
        self.neck = neck
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # Register the hook to automatically convert old version state dicts
        # self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def forward(
        self,
        inputs: torch.Tensor,
        entity: KeypointDetBatchDataEntity | None = None,
        mode: str = "tensor",
    ) -> dict[str, torch.Tensor] | list[InstanceData] | tuple[torch.Tensor] | torch.Tensor:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: 'tensor', 'predict' and 'loss':

        - 'tensor': Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - 'predict': Forward and return the predictions, which are fully
        processed to a list of :obj:`PoseDataSample`.
        - 'loss': Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general
            data_samples (list[:obj:`PoseDataSample`], optional): The
                annotation of every sample. Defaults to ``None``
            mode (str): Set the forward mode and return value type. Defaults
                to ``'tensor'``

        Returns:
            The return type depends on ``mode``.

            - If ``mode='tensor'``, return a tensor or a tuple of tensors
            - If ``mode='predict'``, return a list of :obj:``PoseDataSample``
                that contains the pose predictions
            - If ``mode='loss'``, return a dict of tensor(s) which is the loss
                function value
        """
        if mode == "loss":
            if entity is None:
                msg = "KeypointDetBatchDataEntity should be fed into a model for training."
                raise RuntimeError(msg)
            return self.loss(inputs, entity)
        if mode == "predict":
            return self.predict(inputs)
        if mode == "tensor":
            return self._forward(inputs)

        msg = f'Invalid mode "{mode}". Only supports loss, predict and tensor mode.'
        raise RuntimeError(msg)

    def _forward(
        self,
        inputs: Tensor,
    ) -> Tensor | tuple[Tensor]:
        """Network forward process. Usually includes backbone, neck and head forward without any post-processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).

        Returns:
            Union[Tensor | Tuple[Tensor]]: forward output of the network.
        """
        x = self.extract_feat(inputs)
        return self.head.forward(x)

    def extract_feat(self, inputs: Tensor) -> tuple[Tensor]:
        """Extract features.

        Args:
            inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have various
            resolutions.
        """
        x = self.backbone(inputs)
        if self.neck:
            x = self.neck(x)
        return x

    def loss(self, inputs: torch.Tensor, entity: KeypointDetBatchDataEntity) -> dict | list:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of losses.
        """
        feats = self.extract_feat(inputs)
        return self.head.loss(feats, entity)

    def predict(self, inputs: torch.Tensor) -> list[InstanceData]:
        """Predict results from inputs and data samples with post-processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W)

        Returns:
            list[:obj:`PoseDataSample`]: The pose estimation results of the
            input images. The return value is `PoseDataSample` instances with
            ``pred_instances`` and ``pred_fields``(optional) field , and
            ``pred_instances`` usually contains the following keys:

                - keypoints (Tensor): predicted keypoint coordinates in shape
                    (num_instances, K, D) where K is the keypoint number and D
                    is the keypoint dimension
                - keypoint_scores (Tensor): predicted keypoint scores in shape
                    (num_instances, K)
        """
        feats = self.extract_feat(inputs)
        return self.head.predict(feats)
