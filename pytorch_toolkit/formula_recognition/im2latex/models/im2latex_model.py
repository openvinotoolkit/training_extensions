import torch.nn as nn
import torch
from collections import OrderedDict
from .backbones.resnet import ResNetLikeBackbone, RESNET_BB_LAYERS
from .backbones.original_harvard_bb import Im2LatexBackBone, IM2LATEX_BB_LAYERS
from .text_recognition_heads.attention_based import TextRecognitionHead, HEAD_LAYERS

BB_LAYERS = RESNET_BB_LAYERS
BB_LAYERS.extend(IM2LATEX_BB_LAYERS)


def head_layers_in_key(key):
    for head_layer in HEAD_LAYERS:
        if head_layer in key:
            return True
    return False


def bb_layers_in_key(key):
    for bb_layer in BB_LAYERS:
        if bb_layer in key:
            return True
    return False


class Im2latexModel(nn.Module):
    class Encoder(nn.Module):
        def __init__(self, im2latex_model):
            super().__init__()
            self.model = im2latex_model

        def __call__(self, input_images):
            encoded = self.model.backbone(input_images)
            row_enc_out, hidden, context = self.model.head.encode(encoded)
            hidden, context, init_0 = self.model.head.init_decoder(row_enc_out, hidden, context)
            return row_enc_out, hidden, context, init_0

    class Decoder(nn.Module):

        def __init__(self, im2latex_model):
            super().__init__()
            self.model = im2latex_model

        def __call__(self, hidden, context, output, row_enc_out, tgt):

            return self.model.head.step_decoding(
                hidden, context, output, row_enc_out, tgt)

    def __init__(self, backbone_type, backbone, out_size, head):
        super().__init__()
        self.head = TextRecognitionHead(out_size, head)
        self.backbone_type = backbone_type
        if self.backbone_type == 'resnet':
            self.backbone = ResNetLikeBackbone(backbone)
        else:
            self.backbone = Im2LatexBackBone()

    def forward(self, input_images, formulas=None):
        features = self.backbone(input_images)
        return self.head(features, formulas)

    def load_weights(self, model_path, old_model=False):
        if model_path is not None:
            checkpoint = torch.load(model_path, map_location='cuda')
            checkpoint = OrderedDict((k.replace(
                'module.', '') if 'module.' in k else k, v) for k, v in checkpoint.items())
            # load models trained in previous versions
            if old_model:
                new_checkpoint = OrderedDict()
                for key, value in checkpoint.items():
                    if head_layers_in_key(key):
                        new_checkpoint["head.{}".format(key)] = value
                    elif bb_layers_in_key(key):
                        new_checkpoint[key.replace("cnn_encoder", "backbone")] = value
                    else:
                        raise KeyError("Unrecognized type of layer, could not load the model correctly")
                self.load_state_dict(new_checkpoint)
                return
            self.load_state_dict(checkpoint)

    def get_encoder_wrapper(self, im2latex_model):
        return Im2latexModel.Encoder(im2latex_model)

    def get_decoder_wrapper(self, im2latex_model):
        return Im2latexModel.Decoder(im2latex_model)
