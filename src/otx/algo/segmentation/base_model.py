from torch import nn
import torch
import math
from torch.utils.model_zoo import load_url
import torch.nn.functional as f
from otx.algo.segmentation.losses import create_criterion
from typing import Dict, Any


class BaseSegmNNModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        decode_head: nn.Module,
        criterion_configuration: Dict[str, str | Any] = {"type": "CrossEntropyLoss", "ignore_index": 255},
        pretrained_weights: str | None = None,
    ) -> None:
        """
        Initializes a SegNext model.

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
        self.criterion = create_criterion(**criterion_configuration)
        self.init_weights()

        if pretrained_weights:
            # load pretrained weights
            pretrained_weights = load_url(pretrained_weights)
            self.load_state_dict(pretrained_weights['state_dict'], strict=False)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1.0)
                nn.init.constant_(m.bias, val=0.0)
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, std=math.sqrt(2.0/fan_out), mean=0)

    def forward(self, images, masks):
        enc_feats = self.backbone(images)
        outputs = self.decode_head(enc_feats)
        outputs = f.interpolate(outputs, size=images.size()[-2:], mode='bilinear', align_corners=True)

        if self.training:
            return self.criterion(outputs, masks)

        return outputs
