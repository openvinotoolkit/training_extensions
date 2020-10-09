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
from .text_recognition_heads.attention_based import TextRecognitionHead


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

    def __init__(self, backbone_type, backbone, out_size, head):
        super().__init__()
        self.head = TextRecognitionHead(out_size, **head)
        self.backbone_type = backbone_type
        if self.backbone_type == 'resnet':
            self.backbone = ResNetLikeBackbone(**backbone)
        else:
            self.backbone = Im2LatexBackbone()

    def forward(self, input_images, formulas=None):
        features = self.backbone(input_images)
        return self.head(features, formulas)

    def is_head_layer(self, key):
        head_layers = self.head.state_dict().keys()
        if key in head_layers:
            return True
        return False

    def is_backbone_layer(self, key):
        if 'cnn_encoder' in key:
            return True
        return False

    def load_weights(self, model_path, map_location='cpu'):
        if model_path is None:
            return
        checkpoint = torch.load(model_path, map_location=map_location)
        checkpoint = OrderedDict((k.replace(
            'module.', '') if 'module.' in k else k, v) for k, v in checkpoint.items())
        # load models trained in previous versions
        def is_old_model(checkpoint):
            for k in checkpoint.keys():
                if 'cnn_encoder' in k:
                    return True
            return False
        old_model = is_old_model(checkpoint)
        if not old_model:
            self.load_state_dict(checkpoint)
            return

        new_checkpoint = OrderedDict()
        for key, value in checkpoint.items():
            if self.is_head_layer(key):
                new_checkpoint["head.{}".format(key)] = value
            elif self.is_backbone_layer(key):
                new_checkpoint[key.replace("cnn_encoder", "backbone")] = value
            else:
                raise KeyError("Unrecognized type of layer, could not load the model correctly")
        self.load_state_dict(new_checkpoint)

    def get_encoder_wrapper(self, im2latex_model):
        return Im2latexModel.EncoderWrapper(im2latex_model)

    def get_decoder_wrapper(self, im2latex_model):
        return Im2latexModel.DecoderWrapper(im2latex_model)
