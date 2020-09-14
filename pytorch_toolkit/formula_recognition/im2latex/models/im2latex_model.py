import torch.nn as nn
import torch
from collections import OrderedDict
from .backbones.resnet import ResNetLikeBackbone
from .backbones.original_harvard_bb import Im2LatexBackBone
from .text_recognition_heads.attention_based import TextRecognitionHead


class Im2latexModel(nn.Module):
    def __init__(self, backbone_type, backbone, out_size, head):
        super(Im2latexModel, self).__init__()
        self.head = TextRecognitionHead(out_size, head)
        self.backbone_type = backbone_type
        if 'resnet' in self.backbone_type:
            self.backbone = ResNetLikeBackbone(backbone)
        else:
            self.backbone = Im2LatexBackBone()

    def forward(self, input_images, formulas=None):
        features = self.backbone(input_images)
        return self.head(features, formulas)

    def load_weights(self, model_path):
        if model_path is not None:
            checkpoint = torch.load(model_path, map_location='cuda')
            checkpoint = OrderedDict((k.replace(
                'module.', '') if 'module.' in k else k, v) for k, v in checkpoint.items())
            self.load_state_dict(checkpoint)