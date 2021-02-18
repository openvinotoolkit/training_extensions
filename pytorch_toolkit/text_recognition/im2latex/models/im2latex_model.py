"""
 Copyright (c) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from collections import OrderedDict

import torch
import torch.nn as nn

from .backbones.original_harvard_bb import Im2LatexBackbone
from .backbones.resnet import ResNetLikeBackbone
from .backbones.vgg16 import VGG16Backbone
from .text_recognition_heads.attention_based import AttentionBasedLSTM
from .text_recognition_heads.ctc_lstm_based import LSTMEncoderDecoder

TEXT_REC_HEADS = {
    "AttentionBasedLSTM": AttentionBasedLSTM,
    "LSTMEncoderDecoder": LSTMEncoderDecoder,
}

BACKBONES = {
    'resnet': ResNetLikeBackbone,
    "im2latex": Im2LatexBackbone,
    "VGG": VGG16Backbone
}


class Im2latexModel(nn.Module):
    class EncoderWrapper(nn.Module):
        def __init__(self, im2latex_model):
            super().__init__()
            self.model = im2latex_model

        def forward(self, input_images):
            encoded = self.model.backbone(input_images)
            row_enc_out, hidden, context = self.model.head.encode(encoded)
            hidden, context, init_0 = self.model.head.init_decoder(row_enc_out, hidden, context)
            return row_enc_out, hidden, context, init_0

    class DecoderWrapper(nn.Module):

        def __init__(self, im2latex_model):
            super().__init__()
            self.model = im2latex_model

        def forward(self, hidden, context, output, row_enc_out, tgt):
            return self.model.head.step_decoding(
                hidden, context, output, row_enc_out, tgt)

    def __init__(self, backbone, out_size, head):
        super().__init__()
        bb_out_channels = backbone.get("output_channels", 512)
        head_in_channels = head.get("encoder_input_size", 512)
        assert bb_out_channels == head_in_channels, f"""
        Number of output channels in the backbone ({bb_out_channels}) must be equal
        to the number of input channels in the head ({head_in_channels}) in case last conv
        is disabled
        """
        head_type = head.pop('type', "AttentionBasedLSTM")
        backbone_type = backbone.pop('type', "resnet")
        self.freeze_backbone = backbone.pop("freeze_backbone")
        self.head = TEXT_REC_HEADS[head_type](out_size, **head)
        self.backbone = BACKBONES[backbone_type](**backbone)

    def forward(self, input_images, formulas=None):
        features = self.backbone(input_images)
        return self.head(features, formulas)

    def load_weights(self, model_path, map_location='cpu'):
        if model_path is None:
            return
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=map_location)
        checkpoint = OrderedDict((k.replace('module.', '') if 'module.' in k else k, v) for k, v in checkpoint.items())
        try:
            self.load_state_dict(checkpoint, strict=False)
        except RuntimeError as missing_keys:
            print("""
            Unexpected keys in state_dict.
            Most probably architecture in the config file differs from the architecture of the checkpoint.
            Please, check the config file.
            """)
            raise RuntimeError from missing_keys
        if self.freeze_backbone:
            print("Freeze backbone layers")
            for layer in self.backbone.parameters():
                layer.requires_grad = False

    def get_encoder_wrapper(self, im2latex_model):
        return Im2latexModel.EncoderWrapper(im2latex_model)

    def get_decoder_wrapper(self, im2latex_model):
        return Im2latexModel.DecoderWrapper(im2latex_model)
