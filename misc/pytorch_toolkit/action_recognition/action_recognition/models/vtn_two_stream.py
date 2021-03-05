import torch
from torch import nn

from action_recognition.models.vtn_motion import VideoTransformerMotion

from ..utils import get_fine_tuning_parameters, load_state
from .video_transformer import VideoTransformer


class VideoTransformerTwoStream(nn.Module):
    def __init__(self, embed_size, sequence_size, encoder_name='resnet34', n_classes=400, input_size=224,
                 pretrained=True, motion_path=None, rgb_path=None, mode='rgbdiff', layer_norm=True):
        """Load the pretrained ResNet and replace top fc layer."""
        super().__init__()

        self.rgb_recoder = VideoTransformer(embed_size, sequence_size, encoder_name, n_classes=n_classes,
                                            input_size=input_size, pretrained=pretrained, num_layers=4,
                                            layer_norm=layer_norm)

        self.motion_decoder = VideoTransformerMotion(embed_size, sequence_size, encoder_name, n_classes=n_classes,
                                                     input_size=input_size, pretrained=pretrained, mode=mode,
                                                     layer_norm=layer_norm)

        if motion_path and rgb_path:
            self.load_separate_trained(motion_path, rgb_path)

    def load_separate_trained(self, motion_path, rgb_path):
        print("Loading rgb model from: {}".format(rgb_path))
        rgb_checkpoint = torch.load(rgb_path.as_posix())
        self.rgb_recoder.load_checkpoint(rgb_checkpoint['state_dict'])

        print("Loading motion model from: {}".format(motion_path))
        motion_checkpoint = torch.load(motion_path.as_posix())
        self.motion_decoder.load_checkpoint(motion_checkpoint['state_dict'])

    def forward(self, rgb_clip=None, flow_clip=None):
        """Extract the image feature vectors."""
        logits_rgb = self.rgb_recoder(rgb_clip)
        motion_input = rgb_clip
        if flow_clip is not None:
            motion_input = flow_clip
        logits_motion = self.motion_decoder(motion_input)

        return 0.5 * logits_rgb + 0.5 * logits_motion

    def trainable_parameters(self):
        param_groups = [
            ('trainable', {'re': r''}),
        ]

        return get_fine_tuning_parameters(self, param_groups)
