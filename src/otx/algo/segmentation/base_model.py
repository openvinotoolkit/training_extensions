from __future__ import annotations

from typing import Any

import torch.nn.functional as f
from torch import nn

from otx.algo.segmentation.losses import create_criterion


class BaseSegmNNModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        decode_head: nn.Module,
        criterion_configuration: list[dict[str, str | Any]] = [
            {"type": "CrossEntropyLoss", "params": {"ignore_index": 255}},
        ],
    ) -> None:
        """Initializes a SegNext model.

        Args:
            backbone (MSCAN): The backbone of the model.
            decode_head (LightHamHead): The decode head of the model.
            criterion (Dict[str, Union[str, int]]): The criterion of the model.
                Defaults to {"type": "CrossEntropyLoss", "ignore_index": 255}.
            pretrained_weights (Optional[str]): The path to the pretrained weights.
                Defaults to None.

        Returns:
            None
        """
        super().__init__()

        self.backbone = backbone
        self.decode_head = decode_head
        self.criterions = create_criterion(criterion_configuration)

    def forward(self, images, masks=None, img_metas=None, mode="tensor"):
        enc_feats = self.backbone(images)
        outputs = self.decode_head(enc_feats)

        if mode == "tensor":
            return outputs

        outputs = f.interpolate(outputs, size=images.size()[-2:], mode="bilinear", align_corners=True)
        if mode == "loss":
            if masks is None:
                msg = "The masks must be provided for training."
                raise ValueError(msg)
            output_losses = {}
            for criterion in self.criterions:
                output_losses.update({criterion.name: criterion(outputs, masks, img_metas=img_metas)})
            return output_losses

        if mode == "predict":
            return outputs.argmax(dim=1)

        return outputs
