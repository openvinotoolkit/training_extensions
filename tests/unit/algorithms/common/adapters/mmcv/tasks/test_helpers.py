"""Test helpers for MPA."""
import torch
import torch.nn as nn


def generate_random_torch_image(batch=1, width=3, height=3, channels=3, channel_last=False):
    """Generate random torch tensor image.

    Args:
        batch (int, optional): A size of batch. Defaults to 1.
        width (int, optional): the image width. Defaults to 224.
        height (int, optional): the image height. Defaults to 224.
        channels (int, optional): the image channel. Defaults to 3.
        channel_last (bool, optional): if this is True, image shape will follow BHWC. Defaults to True.

    Returns:
        _type_: _description_
    """
    if channel_last is False:
        img = torch.rand(batch, channels, height, width)
    else:
        img = torch.rand(batch, height, width, channels)
    return img


def generate_toy_cnn_model(in_channels=3, mid_channels=3, out_channels=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, (1, 1)), nn.BatchNorm2d(mid_channels), nn.AdaptiveAvgPool2d((1, 1))
    )


def generate_toy_head(in_features, out_features):
    return nn.Linear(in_features, out_features)
