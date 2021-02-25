import torch
from torch import nn

from ..utils import get_fine_tuning_parameters, load_state
from .video_transformer import VideoTransformer


class RGBDiff(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, image):
        """
        Args:
            image (torch.Tensor):  (N x T x C x H x W)

        """
        diffs = []
        for i in range(1, image.size(self.dim)):
            prev = image.index_select(self.dim, image.new_tensor(i - 1, dtype=torch.long))
            current = image.index_select(self.dim, image.new_tensor(i, dtype=torch.long))
            diffs.append(current - prev)

        return torch.cat(diffs, dim=self.dim)


class VideoTransformerMotion(nn.Module):
    def __init__(self, embed_size, sequence_size, encoder_name, n_classes=400, input_size=224, pretrained=True,
                 mode='rfbdiff', layer_norm=True):
        """Load the pretrained ResNet and replace top fc layer."""
        super().__init__()
        self.mode = mode
        motion_sequence_size = sequence_size
        input_channels = 3
        if self.mode == "flow":
            input_channels = 2
        elif self.mode == "rgbdiff":
            motion_sequence_size = motion_sequence_size - 1
            self.rgb_diff = RGBDiff()
        else:
            raise Exception("Unsupported mode " + self.mode)

        self.motion_decoder = VideoTransformer(embed_size, motion_sequence_size, encoder_name, n_classes=n_classes,
                                               input_size=input_size, pretrained=pretrained,
                                               input_channels=input_channels, layer_norm=layer_norm)

    def forward(self, clip):
        """Extract the image feature vectors."""
        if self.mode == "rgbdiff":
            clip = self.rgb_diff(clip)
        logits_motion = self.motion_decoder(clip)

        return logits_motion

    def trainable_parameters(self):
        param_groups = [
            ('trainable', {'re': r''}),
        ]

        return get_fine_tuning_parameters(self, param_groups)

    def load_checkpoint(self, state_dict):
        load_state(self, state_dict, 'motion_decoder.fc')
