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

from .backbones.resnet import (CustomResNetLikeBackbone, ResNetLikeBackbone)
from .text_recognition_heads.attention_based import AttentionBasedLSTM
from .text_recognition_heads.attention_based_2d import \
    TextRecognitionHeadAttention
from .text_recognition_heads.ctc_lstm_based import LSTMEncoderDecoder
from .transformation.tps import TPS_SpatialTransformerNetwork

TEXT_REC_HEADS = {
    'AttentionBasedLSTM': AttentionBasedLSTM,
    'LSTMEncoderDecoder': LSTMEncoderDecoder,
    'TextRecognitionHeadAttention': TextRecognitionHeadAttention,
}

BACKBONES = {
    'resnet': ResNetLikeBackbone,
    'custom_resnet': CustomResNetLikeBackbone,
}

TRANSFORMATIONS = {
    'tps': TPS_SpatialTransformerNetwork,
}


class TextRecognitionModel(nn.Module):
    class EncoderWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_images):
            if hasattr(self.model, 'transformation'):
                input_images = self.model.transformation(input_images)
            encoded = self.model.backbone(input_images)
            return self.model.head.encoder_wrapper(encoded)

    class DecoderWrapper(nn.Module):

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, args):
            return self.model.head.decoder_wrapper(*args)

    def __init__(self, backbone, out_size, head, transformation):
        super().__init__()
        bb_out_channels = backbone.get('output_channels', 512)
        head_in_channels = head.get('encoder_input_size', 512)
        assert bb_out_channels == head_in_channels, f"""
        Number of output channels in the backbone ({bb_out_channels}) must be equal
        to the number of input channels in the head ({head_in_channels}) in case last conv
        is disabled
        """
        head_type = head.pop('type', 'AttentionBasedLSTM')
        backbone_type = backbone.pop('type', 'resnet')
        transformation_type = transformation.pop('type', None)
        if transformation_type:
            self.transformation = TRANSFORMATIONS[transformation_type](**transformation)
        self.freeze_backbone = backbone.pop('freeze_backbone', False)
        self.head = TEXT_REC_HEADS[head_type](out_size, **head)
        self.backbone = BACKBONES[backbone_type](**backbone)
        if self.freeze_backbone:
            print('Freeze backbone layers')
            for layer in self.backbone.parameters():
                layer.requires_grad = False
            if backbone.get('one_ch_first_conv'):
                for layer in self.backbone.conv1.parameters():
                    layer.requires_grad = True

    def forward(self, input_images, formulas=None):
        if hasattr(self, 'transformation'):
            input_images = self.transformation(input_images)
        features = self.backbone(input_images)
        return self.head(features, formulas)

    def load_weights(self, model_path, map_location='cpu'):
        if model_path is None:
            return
        print(f'Loading model from {model_path}')
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

    def get_encoder_wrapper(self, model):
        return TextRecognitionModel.EncoderWrapper(model)

    def get_decoder_wrapper(self, model):
        return TextRecognitionModel.DecoderWrapper(model)
