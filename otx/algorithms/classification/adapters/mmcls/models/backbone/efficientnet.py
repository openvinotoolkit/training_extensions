# Copyright (c) 2018-2022 Kaiyang Zhou
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""
    EfficientNet for ImageNet-1K, implemented in PyTorch.
    Original papers:
    - 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,' https://arxiv.org/abs/1905.11946,
    - 'Adversarial Examples Improve Image Recognition,' https://arxiv.org/abs/1911.09665.
"""

import math
import os

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from mmcv.cnn import build_activation_layer
from mmcv.cnn.bricks import ConvModule
from mmcv.runner import load_checkpoint

from mmcls.models.builder import BACKBONES

from mpa.utils.logger import get_logger

logger = get_logger()

pretrained_root = "https://github.com/osmr/imgclsmob/releases/download/v0.0.364/"
pretrained_urls = {
    "efficientnet_b0": pretrained_root + "efficientnet_b0-0752-0e386130.pth.zip",
}


def conv1x1_block(
    in_channels,
    out_channels,
    stride=1,
    padding=0,
    groups=1,
    bias=False,
    use_bn=True,
    bn_eps=1e-5,
    activation="ReLU",
):
    return ConvModule(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        norm_cfg=(dict(type="BN", eps=bn_eps) if use_bn else None),
        act_cfg=(dict(type=activation) if activation else None),
    )


def conv3x3_block(
    in_channels,
    out_channels,
    stride=1,
    padding=1,
    dilation=1,
    groups=1,
    bias=False,
    use_bn=True,
    bn_eps=1e-5,
    activation="ReLU",
    IN_conv=False,
):
    return ConvModule(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        norm_cfg=(dict(type="BN", eps=bn_eps) if use_bn else None),
        act_cfg=(dict(type=activation) if activation else None),
    )


def dwconv3x3_block(
    in_channels,
    out_channels,
    stride=1,
    padding=1,
    dilation=1,
    bias=False,
    use_bn=True,
    bn_eps=1e-5,
    activation="ReLU",
):
    return ConvModule(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=out_channels,
        bias=bias,
        norm_cfg=(dict(type="BN", eps=bn_eps) if use_bn else None),
        act_cfg=(dict(type=activation) if activation else None),
    )


def dwconv5x5_block(
    in_channels,
    out_channels,
    stride=1,
    padding=2,
    dilation=1,
    bias=False,
    use_bn=True,
    bn_eps=1e-5,
    activation="ReLU",
):
    return ConvModule(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=5,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=out_channels,
        bias=bias,
        norm_cfg=(dict(type="BN", eps=bn_eps) if use_bn else None),
        act_cfg=(dict(type=activation) if activation else None),
    )


def round_channels(channels, divisor=8):
    """
    Round weighted channel number (make divisible operation).
    Parameters:
    ----------
    channels : int or float
        Original number of channels.
    divisor : int, default 8
        Alignment value.
    Returns
    -------
    int
        Weighted number of channels.
    """
    rounded_channels = max(int(channels + divisor / 2.0) // divisor * divisor, divisor)
    if float(rounded_channels) < 0.9 * channels:
        rounded_channels += divisor
    return rounded_channels


def calc_tf_padding(x, kernel_size, stride=1, dilation=1):
    """
    Calculate TF-same like padding size.
    Parameters:
    ----------
    x : tensor
        Input tensor.
    kernel_size : int
        Convolution window size.
    stride : int, default 1
        Strides of the convolution.
    dilation : int, default 1
        Dilation value for convolution layer.
    Returns
    -------
    tuple of 4 int
        The size of the padding.
    """
    height, width = x.size()[2:]
    oh = math.ceil(height / stride)
    ow = math.ceil(width / stride)
    pad_h = max((oh - 1) * stride + (kernel_size - 1) * dilation + 1 - height, 0)
    pad_w = max((ow - 1) * stride + (kernel_size - 1) * dilation + 1 - width, 0)
    return pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    Parameters:
    ----------
    channels : int
        Number of channels.
    reduction : int, default 16
        Squeeze reduction value.
    mid_channels : int or None, default None
        Number of middle channels.
    round_mid : bool, default False
        Whether to round middle channel number (make divisible by 8).
    use_conv : bool, default True
        Whether to convolutional layers instead of fully-connected ones.
    activation : function, or str, or nn.Module, default 'relu'
        Activation function after the first convolution.
    out_activation : function, or str, or nn.Module, default 'sigmoid'
        Activation function after the last convolution.
    """

    def __init__(
        self,
        channels,
        reduction=16,
        mid_channels=None,
        round_mid=False,
        use_conv=True,
        mid_activation="ReLU",
        out_activation="Sigmoid",
    ):
        super(SEBlock, self).__init__()
        self.use_conv = use_conv
        if mid_channels is None:
            mid_channels = (
                channels // reduction
                if not round_mid
                else round_channels(float(channels) / reduction)
            )

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        if use_conv:
            self.conv1 = nn.Conv2d(
                in_channels=channels,
                out_channels=mid_channels,
                kernel_size=1,
                stride=1,
                groups=1,
                bias=True,
            )
        else:
            self.fc1 = nn.Linear(in_features=channels, out_features=mid_channels)
        self.activ = build_activation_layer(dict(type=mid_activation))
        if use_conv:
            self.conv2 = nn.Conv2d(
                in_channels=mid_channels,
                out_channels=channels,
                kernel_size=1,
                stride=1,
                groups=1,
                bias=True,
            )
        else:
            self.fc2 = nn.Linear(in_features=mid_channels, out_features=channels)
        self.sigmoid = build_activation_layer(dict(type=out_activation))

    def forward(self, x):
        w = self.pool(x)
        if not self.use_conv:
            w = w.view(x.size(0), -1)
        w = self.conv1(w) if self.use_conv else self.fc1(w)
        w = self.activ(w)
        w = self.conv2(w) if self.use_conv else self.fc2(w)
        w = self.sigmoid(w)
        if not self.use_conv:
            w = w.unsqueeze(2).unsqueeze(3)
        x = x * w
        return x


class EffiDwsConvUnit(nn.Module):
    """
    EfficientNet specific depthwise separable convolution block/unit with BatchNorms and activations at each convolution
    layers.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the second convolution layer.
    bn_eps : float
        Small float added to variance in Batch norm.
    activation : str
        Name of activation function.
    tf_mode : bool
        Whether to use TF-like mode.
    """

    def __init__(self, in_channels, out_channels, stride, bn_eps, activation, tf_mode):
        super(EffiDwsConvUnit, self).__init__()
        self.tf_mode = tf_mode
        self.residual = (in_channels == out_channels) and (stride == 1)

        self.dw_conv = dwconv3x3_block(
            in_channels=in_channels,
            out_channels=in_channels,
            padding=(0 if tf_mode else 1),
            bn_eps=bn_eps,
            activation=activation,
        )
        self.se = SEBlock(channels=in_channels, reduction=4, mid_activation=activation)
        self.pw_conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            bn_eps=bn_eps,
            activation=None,
        )

    def forward(self, x):
        if self.residual:
            identity = x
        if self.tf_mode:
            x = F.pad(x, pad=calc_tf_padding(x, kernel_size=3))
        x = self.dw_conv(x)
        x = self.se(x)
        x = self.pw_conv(x)
        if self.residual:
            x = x + identity
        return x


class EffiInvResUnit(nn.Module):
    """
    EfficientNet inverted residual unit.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the second convolution layer.
    exp_factor : int
        Factor for expansion of channels.
    se_factor : int
        SE reduction factor for each unit.
    bn_eps : float
        Small float added to variance in Batch norm.
    activation : str
        Name of activation function.
    tf_mode : bool
        Whether to use TF-like mode.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        exp_factor,
        se_factor,
        bn_eps,
        activation,
        tf_mode,
    ):
        super(EffiInvResUnit, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.tf_mode = tf_mode
        self.residual = (in_channels == out_channels) and (stride == 1)
        self.use_se = se_factor > 0
        mid_channels = in_channels * exp_factor
        dwconv_block_fn = (
            dwconv3x3_block
            if kernel_size == 3
            else (dwconv5x5_block if kernel_size == 5 else None)
        )

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            bn_eps=bn_eps,
            activation=activation,
        )
        self.conv2 = dwconv_block_fn(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride,
            padding=(0 if tf_mode else (kernel_size // 2)),
            bn_eps=bn_eps,
            activation=activation,
        )
        if self.use_se:
            self.se = SEBlock(
                channels=mid_channels,
                reduction=(exp_factor * se_factor),
                mid_activation=activation,
            )
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            bn_eps=bn_eps,
            activation=None,
        )

    def forward(self, x):
        if self.residual:
            identity = x
        x = self.conv1(x)
        if self.tf_mode:
            x = F.pad(
                x,
                pad=calc_tf_padding(
                    x, kernel_size=self.kernel_size, stride=self.stride
                ),
            )
        x = self.conv2(x)
        if self.use_se:
            x = self.se(x)
        x = self.conv3(x)
        if self.residual:
            x = x + identity
        return x


class EffiInitBlock(nn.Module):
    """
    EfficientNet specific initial block.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_eps : float
        Small float added to variance in Batch norm.
    activation : str
        Name of activation function.
    tf_mode : bool
        Whether to use TF-like mode.
    """

    def __init__(
        self, in_channels, out_channels, bn_eps, activation, tf_mode, IN_conv1
    ):
        super(EffiInitBlock, self).__init__()
        self.tf_mode = tf_mode

        self.conv = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            padding=(0 if tf_mode else 1),
            bn_eps=bn_eps,
            activation=activation,
            IN_conv=IN_conv1,
        )

    def forward(self, x):
        if self.tf_mode:
            x = F.pad(x, pad=calc_tf_padding(x, kernel_size=3, stride=2))
        x = self.conv(x)
        return x


class EfficientNet(nn.Module):
    """
    EfficientNet model from 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,'
    https://arxiv.org/abs/1905.11946.
    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    kernel_sizes : list of list of int
        Number of kernel sizes for each unit.
    strides_per_stage : list int
        Stride value for the first unit of each stage.
    expansion_factors : list of list of int
        Number of expansion factors for each unit.
    dropout_rate : float, default 0.2
        Fraction of the input units to drop. Must be a number between 0 and 1.
    tf_mode : bool, default False
        Whether to use TF-like mode.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 final_block_channels,
                 kernel_sizes,
                 strides_per_stage,
                 expansion_factors,
                 tf_mode=False,
                 bn_eps=1e-5,
                 in_channels=3,
                 in_size=(224, 224),
                 dropout_cls=None,
                 pooling_type='avg',
                 bn_eval=False,
                 bn_frozen=False,
                 IN_first=False,
                 IN_conv1=False,
                 pretrained=False,
                 **kwargs):

        super().__init__(**kwargs)
        self.num_classes = 1000
        self.pretrained = pretrained
        self.in_size = in_size
        self.input_IN = nn.InstanceNorm2d(3, affine=True) if IN_first else None
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.pooling_type = pooling_type
        self.num_features = self.num_head_features = final_block_channels
        activation = 'Swish'
        self.features = nn.Sequential()
        self.features.add_module("init_block", EffiInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            bn_eps=bn_eps,
            activation=activation,
            tf_mode=tf_mode,
            IN_conv1=IN_conv1))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            kernel_sizes_per_stage = kernel_sizes[i]
            expansion_factors_per_stage = expansion_factors[i]
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                kernel_size = kernel_sizes_per_stage[j]
                expansion_factor = expansion_factors_per_stage[j]
                stride = strides_per_stage[i] if (j == 0) else 1
                if i == 0:
                    stage.add_module("unit{}".format(j + 1), EffiDwsConvUnit(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=stride,
                        bn_eps=bn_eps,
                        activation=activation,
                        tf_mode=tf_mode))
                else:
                    stage.add_module("unit{}".format(j + 1), EffiInvResUnit(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        exp_factor=expansion_factor,
                        se_factor=4,
                        bn_eps=bn_eps,
                        activation=activation,
                        tf_mode=tf_mode))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
            # activation = activation if self.loss == 'softmax': else lambda: nn.PReLU(init=0.25)
        self.features.add_module("final_block", conv1x1_block(
            in_channels=in_channels,
            out_channels=final_block_channels,
            bn_eps=bn_eps,
            activation=activation))

        """ Comment out unused part. Only use 'backbone' part in mpa.
        self.output = nn.Sequential()
        if dropout_cls:
            self.output.add_module("dropout", Dropout(**dropout_cls))
        if self.loss in ['softmax', 'asl']:
            self.output.add_module("fc", nn.Linear(
                in_features=final_block_channels,
                out_features=self.num_classes))
        else:
            assert self.loss in ['am_softmax', 'am_binary']
            self.output.add_module("asl", AngleSimpleLinear(
                in_features=final_block_channels,
                out_features=self.num_classes))
        """

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x, return_featuremaps=False, get_embeddings=False):
        if self.input_IN is not None:
            x = self.input_IN(x)

        y = self.features(x)
        if return_featuremaps:
            return y

        glob_features = self._glob_feature_vector(
            y, self.pooling_type, reduce_dims=False
        )

        logits = self.output(glob_features.view(x.shape[0], -1))

        if not self.training and self.classification:
            return [logits]

        if get_embeddings:
            out_data = [logits, glob_features.view(x.shape[0], -1)]
        elif self.loss in ["softmax", "am_softmax"]:
            if self.lr_finder.enable and self.lr_finder.mode == "automatic":
                out_data = logits
            else:
                out_data = [logits]

        elif self.loss in ["triplet"]:
            out_data = [logits, glob_features]
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

        if self.lr_finder.enable and self.lr_finder.mode == "automatic":
            return out_data
        return tuple(out_data)


def get_efficientnet(
    version,
    in_size,
    tf_mode=False,
    bn_eps=1e-5,
    model_name=None,
    pretrained=False,
    root=os.path.join("~", ".torch", "models"),
    **kwargs,
):
    """
    Create EfficientNet model with specific parameters.
    Parameters:
    ----------
    version : str
        Version of EfficientNet ('b0'...'b8').
    in_size : tuple of two ints
        Spatial size of the expected input image.
    tf_mode : bool, default False
        Whether to use TF-like mode.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    if version == "b0":
        assert in_size == (224, 224)
        depth_factor = 1.0
        width_factor = 1.0
        dropout_rate = 0.2
    elif version == "b1":
        assert in_size == (240, 240)
        depth_factor = 1.1
        width_factor = 1.0
        dropout_rate = 0.2
    elif version == "b2":
        assert in_size == (260, 260)
        depth_factor = 1.2
        width_factor = 1.1
        dropout_rate = 0.3
    elif version == "b3":
        assert in_size == (300, 300)
        depth_factor = 1.4
        width_factor = 1.2
        dropout_rate = 0.3
    elif version == "b4":
        assert in_size == (380, 380)
        depth_factor = 1.8
        width_factor = 1.4
        dropout_rate = 0.4
    elif version == "b5":
        assert in_size == (456, 456)
        depth_factor = 2.2
        width_factor = 1.6
        dropout_rate = 0.4
    elif version == "b6":
        assert in_size == (528, 528)
        depth_factor = 2.6
        width_factor = 1.8
        dropout_rate = 0.5
    elif version == "b7":
        assert in_size == (600, 600)
        depth_factor = 3.1
        width_factor = 2.0
        dropout_rate = 0.5
    elif version == "b8":
        assert in_size == (672, 672)
        depth_factor = 3.6
        width_factor = 2.2
        dropout_rate = 0.5
    else:
        raise ValueError("Unsupported EfficientNet version {}".format(version))

    init_block_channels = 32
    layers = [1, 2, 2, 3, 3, 4, 1]
    downsample = [1, 1, 1, 1, 0, 1, 0]
    channels_per_layers = [16, 24, 40, 80, 112, 192, 320]
    expansion_factors_per_layers = [1, 6, 6, 6, 6, 6, 6]
    kernel_sizes_per_layers = [3, 3, 5, 3, 5, 5, 3]
    strides_per_stage = [1, 2, 2, 2, 1, 2, 1]
    final_block_channels = 1280

    layers = [int(math.ceil(li * depth_factor)) for li in layers]
    channels_per_layers = [
        round_channels(ci * width_factor) for ci in channels_per_layers
    ]

    from functools import reduce

    channels = reduce(
        lambda x, y: x + [[y[0]] * y[1]]
        if y[2] != 0
        else x[:-1] + [x[-1] + [y[0]] * y[1]],
        zip(channels_per_layers, layers, downsample),
        [],
    )
    kernel_sizes = reduce(
        lambda x, y: x + [[y[0]] * y[1]]
        if y[2] != 0
        else x[:-1] + [x[-1] + [y[0]] * y[1]],
        zip(kernel_sizes_per_layers, layers, downsample),
        [],
    )
    expansion_factors = reduce(
        lambda x, y: x + [[y[0]] * y[1]]
        if y[2] != 0
        else x[:-1] + [x[-1] + [y[0]] * y[1]],
        zip(expansion_factors_per_layers, layers, downsample),
        [],
    )
    strides_per_stage = reduce(
        lambda x, y: x + [[y[0]] * y[1]]
        if y[2] != 0
        else x[:-1] + [x[-1] + [y[0]] * y[1]],
        zip(strides_per_stage, layers, downsample),
        [],
    )
    strides_per_stage = [si[0] for si in strides_per_stage]

    init_block_channels = round_channels(init_block_channels * width_factor)

    if width_factor > 1.0:
        assert int(final_block_channels * width_factor) == round_channels(
            final_block_channels * width_factor
        )
        final_block_channels = round_channels(final_block_channels * width_factor)

    net = EfficientNet(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        kernel_sizes=kernel_sizes,
        strides_per_stage=strides_per_stage,
        expansion_factors=expansion_factors,
        dropout_rate=dropout_rate,
        tf_mode=tf_mode,
        bn_eps=bn_eps,
        in_size=in_size,
        **kwargs,
    )

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError(
                "Parameter `model_name` should be properly initialized for loading pretrained model."
            )
        from .model_store import download_model

        download_model(net=net, model_name=model_name, local_model_store_dir_path=root)

    return net


def efficientnet_b0(in_size=(224, 224), **kwargs):
    """
    EfficientNet-B0 model from 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,'
    https://arxiv.org/abs/1905.11946.
    Parameters:
    ----------
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_efficientnet(
        version="b0", in_size=in_size, model_name="efficientnet_b0", **kwargs
    )


def efficientnet_b1(in_size=(240, 240), **kwargs):
    """
    EfficientNet-B1 model from 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,'
    https://arxiv.org/abs/1905.11946.
    Parameters:
    ----------
    in_size : tuple of two ints, default (240, 240)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_efficientnet(
        version="b1", in_size=in_size, model_name="efficientnet_b1", **kwargs
    )


def efficientnet_b2(in_size=(260, 260), **kwargs):
    """
    EfficientNet-B2 model from 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,'
    https://arxiv.org/abs/1905.11946.
    Parameters:
    ----------
    in_size : tuple of two ints, default (260, 260)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_efficientnet(
        version="b2", in_size=in_size, model_name="efficientnet_b2", **kwargs
    )


def efficientnet_b3(in_size=(300, 300), **kwargs):
    """
    EfficientNet-B3 model from 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,'
    https://arxiv.org/abs/1905.11946.
    Parameters:
    ----------
    in_size : tuple of two ints, default (300, 300)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_efficientnet(
        version="b3", in_size=in_size, model_name="efficientnet_b3", **kwargs
    )


def efficientnet_b4(in_size=(380, 380), **kwargs):
    """
    EfficientNet-B4 model from 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,'
    https://arxiv.org/abs/1905.11946.
    Parameters:
    ----------
    in_size : tuple of two ints, default (380, 380)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_efficientnet(
        version="b4", in_size=in_size, model_name="efficientnet_b4", **kwargs
    )


def efficientnet_b5(in_size=(456, 456), **kwargs):
    """
    EfficientNet-B5 model from 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,'
    https://arxiv.org/abs/1905.11946.
    Parameters:
    ----------
    in_size : tuple of two ints, default (456, 456)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_efficientnet(
        version="b5", in_size=in_size, model_name="efficientnet_b5", **kwargs
    )


def efficientnet_b6(in_size=(528, 528), **kwargs):
    """
    EfficientNet-B6 model from 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,'
    https://arxiv.org/abs/1905.11946.
    Parameters:
    ----------
    in_size : tuple of two ints, default (528, 528)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_efficientnet(
        version="b6", in_size=in_size, model_name="efficientnet_b6", **kwargs
    )


def efficientnet_b7(in_size=(600, 600), **kwargs):
    """
    EfficientNet-B7 model from 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,'
    https://arxiv.org/abs/1905.11946.
    Parameters:
    ----------
    in_size : tuple of two ints, default (600, 600)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_efficientnet(
        version="b7", in_size=in_size, model_name="efficientnet_b7", **kwargs
    )


def efficientnet_b8(in_size=(672, 672), **kwargs):
    """
    EfficientNet-B8 model from 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,'
    https://arxiv.org/abs/1905.11946.
    Parameters:
    ----------
    in_size : tuple of two ints, default (672, 672)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_efficientnet(
        version="b8", in_size=in_size, model_name="efficientnet_b8", **kwargs
    )


def efficientnet_b0b(in_size=(224, 224), **kwargs):
    """
    EfficientNet-B0-b (like TF-implementation) model from 'EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks,' https://arxiv.org/abs/1905.11946.
    Parameters:
    ----------
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_efficientnet(
        version="b0",
        in_size=in_size,
        tf_mode=True,
        bn_eps=1e-3,
        model_name="efficientnet_b0b",
        **kwargs,
    )


def efficientnet_b1b(in_size=(240, 240), **kwargs):
    """
    EfficientNet-B1-b (like TF-implementation) model from 'EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks,' https://arxiv.org/abs/1905.11946.
    Parameters:
    ----------
    in_size : tuple of two ints, default (240, 240)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_efficientnet(
        version="b1",
        in_size=in_size,
        tf_mode=True,
        bn_eps=1e-3,
        model_name="efficientnet_b1b",
        **kwargs,
    )


def efficientnet_b2b(in_size=(260, 260), **kwargs):
    """
    EfficientNet-B2-b (like TF-implementation) model from 'EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks,' https://arxiv.org/abs/1905.11946.
    Parameters:
    ----------
    in_size : tuple of two ints, default (260, 260)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_efficientnet(
        version="b2",
        in_size=in_size,
        tf_mode=True,
        bn_eps=1e-3,
        model_name="efficientnet_b2b",
        **kwargs,
    )


def efficientnet_b3b(in_size=(300, 300), **kwargs):
    """
    EfficientNet-B3-b (like TF-implementation) model from 'EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks,' https://arxiv.org/abs/1905.11946.
    Parameters:
    ----------
    in_size : tuple of two ints, default (300, 300)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_efficientnet(
        version="b3",
        in_size=in_size,
        tf_mode=True,
        bn_eps=1e-3,
        model_name="efficientnet_b3b",
        **kwargs,
    )


def efficientnet_b4b(in_size=(380, 380), **kwargs):
    """
    EfficientNet-B4-b (like TF-implementation) model from 'EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks,' https://arxiv.org/abs/1905.11946.
    Parameters:
    ----------
    in_size : tuple of two ints, default (380, 380)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_efficientnet(
        version="b4",
        in_size=in_size,
        tf_mode=True,
        bn_eps=1e-3,
        model_name="efficientnet_b4b",
        **kwargs,
    )


def efficientnet_b5b(in_size=(456, 456), **kwargs):
    """
    EfficientNet-B5-b (like TF-implementation) model from 'EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks,' https://arxiv.org/abs/1905.11946.
    Parameters:
    ----------
    in_size : tuple of two ints, default (456, 456)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_efficientnet(
        version="b5",
        in_size=in_size,
        tf_mode=True,
        bn_eps=1e-3,
        model_name="efficientnet_b5b",
        **kwargs,
    )


def efficientnet_b6b(in_size=(528, 528), **kwargs):
    """
    EfficientNet-B6-b (like TF-implementation) model from 'EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks,' https://arxiv.org/abs/1905.11946.
    Parameters:
    ----------
    in_size : tuple of two ints, default (528, 528)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_efficientnet(
        version="b6",
        in_size=in_size,
        tf_mode=True,
        bn_eps=1e-3,
        model_name="efficientnet_b6b",
        **kwargs,
    )


def efficientnet_b7b(in_size=(600, 600), **kwargs):
    """
    EfficientNet-B7-b (like TF-implementation) model from 'EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks,' https://arxiv.org/abs/1905.11946.
    Parameters:
    ----------
    in_size : tuple of two ints, default (600, 600)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_efficientnet(
        version="b7",
        in_size=in_size,
        tf_mode=True,
        bn_eps=1e-3,
        model_name="efficientnet_b7b",
        **kwargs,
    )


def efficientnet_b0c(in_size=(224, 224), **kwargs):
    """
    EfficientNet-B0-c (like TF-implementation, trained with AdvProp) model from 'EfficientNet: Rethinking Model Scaling
    for Convolutional Neural Networks,' https://arxiv.org/abs/1905.11946.
    Parameters:
    ----------
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_efficientnet(
        version="b0",
        in_size=in_size,
        tf_mode=True,
        bn_eps=1e-3,
        model_name="efficientnet_b0c",
        **kwargs,
    )


def efficientnet_b1c(in_size=(240, 240), **kwargs):
    """
    EfficientNet-B1-c (like TF-implementation, trained with AdvProp) model from 'EfficientNet: Rethinking Model Scaling
    for Convolutional Neural Networks,' https://arxiv.org/abs/1905.11946.
    Parameters:
    ----------
    in_size : tuple of two ints, default (240, 240)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_efficientnet(
        version="b1",
        in_size=in_size,
        tf_mode=True,
        bn_eps=1e-3,
        model_name="efficientnet_b1c",
        **kwargs,
    )


def efficientnet_b2c(in_size=(260, 260), **kwargs):
    """
    EfficientNet-B2-c (like TF-implementation, trained with AdvProp) model from 'EfficientNet: Rethinking Model Scaling
    for Convolutional Neural Networks,' https://arxiv.org/abs/1905.11946.
    Parameters:
    ----------
    in_size : tuple of two ints, default (260, 260)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_efficientnet(
        version="b2",
        in_size=in_size,
        tf_mode=True,
        bn_eps=1e-3,
        model_name="efficientnet_b2c",
        **kwargs,
    )


def efficientnet_b3c(in_size=(300, 300), **kwargs):
    """
    EfficientNet-B3-c (like TF-implementation, trained with AdvProp) model from 'EfficientNet: Rethinking Model Scaling
    for Convolutional Neural Networks,' https://arxiv.org/abs/1905.11946.
    Parameters:
    ----------
    in_size : tuple of two ints, default (300, 300)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_efficientnet(
        version="b3",
        in_size=in_size,
        tf_mode=True,
        bn_eps=1e-3,
        model_name="efficientnet_b3c",
        **kwargs,
    )


def efficientnet_b4c(in_size=(380, 380), **kwargs):
    """
    EfficientNet-B4-c (like TF-implementation, trained with AdvProp) model from 'EfficientNet: Rethinking Model Scaling
    for Convolutional Neural Networks,' https://arxiv.org/abs/1905.11946.
    Parameters:
    ----------
    in_size : tuple of two ints, default (380, 380)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_efficientnet(
        version="b4",
        in_size=in_size,
        tf_mode=True,
        bn_eps=1e-3,
        model_name="efficientnet_b4c",
        **kwargs,
    )


def efficientnet_b5c(in_size=(456, 456), **kwargs):
    """
    EfficientNet-B5-c (like TF-implementation, trained with AdvProp) model from 'EfficientNet: Rethinking Model Scaling
    for Convolutional Neural Networks,' https://arxiv.org/abs/1905.11946.
    Parameters:
    ----------
    in_size : tuple of two ints, default (456, 456)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_efficientnet(
        version="b5",
        in_size=in_size,
        tf_mode=True,
        bn_eps=1e-3,
        model_name="efficientnet_b5c",
        **kwargs,
    )


def efficientnet_b6c(in_size=(528, 528), **kwargs):
    """
    EfficientNet-B6-c (like TF-implementation, trained with AdvProp) model from 'EfficientNet: Rethinking Model Scaling
    for Convolutional Neural Networks,' https://arxiv.org/abs/1905.11946.
    Parameters:
    ----------
    in_size : tuple of two ints, default (528, 528)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_efficientnet(
        version="b6",
        in_size=in_size,
        tf_mode=True,
        bn_eps=1e-3,
        model_name="efficientnet_b6c",
        **kwargs,
    )


def efficientnet_b7c(in_size=(600, 600), **kwargs):
    """
    EfficientNet-B7-c (like TF-implementation, trained with AdvProp) model from 'EfficientNet: Rethinking Model Scaling
    for Convolutional Neural Networks,' https://arxiv.org/abs/1905.11946.
    Parameters:
    ----------
    in_size : tuple of two ints, default (600, 600)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_efficientnet(
        version="b7",
        in_size=in_size,
        tf_mode=True,
        bn_eps=1e-3,
        model_name="efficientnet_b7c",
        **kwargs,
    )


def efficientnet_b8c(in_size=(672, 672), **kwargs):
    """
    EfficientNet-B8-c (like TF-implementation, trained with AdvProp) model from 'EfficientNet: Rethinking Model Scaling
    for Convolutional Neural Networks,' https://arxiv.org/abs/1905.11946.
    Parameters:
    ----------
    in_size : tuple of two ints, default (672, 672)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_efficientnet(
        version="b8",
        in_size=in_size,
        tf_mode=True,
        bn_eps=1e-3,
        model_name="efficientnet_b8c",
        **kwargs,
    )


def _calc_width(net):
    import numpy as np

    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import torch

    pretrained = False

    models = [
        efficientnet_b0,
        efficientnet_b1,
        efficientnet_b2,
        efficientnet_b3,
        efficientnet_b4,
        efficientnet_b5,
        efficientnet_b6,
        efficientnet_b7,
        efficientnet_b8,
        efficientnet_b0b,
        efficientnet_b1b,
        efficientnet_b2b,
        efficientnet_b3b,
        efficientnet_b4b,
        efficientnet_b5b,
        efficientnet_b6b,
        efficientnet_b7b,
        efficientnet_b0c,
        efficientnet_b1c,
        efficientnet_b2c,
        efficientnet_b3c,
        efficientnet_b4c,
        efficientnet_b5c,
        efficientnet_b6c,
        efficientnet_b7c,
        efficientnet_b8c,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert model != efficientnet_b0 or weight_count == 5288548
        assert model != efficientnet_b1 or weight_count == 7794184
        assert model != efficientnet_b2 or weight_count == 9109994
        assert model != efficientnet_b3 or weight_count == 12233232
        assert model != efficientnet_b4 or weight_count == 19341616
        assert model != efficientnet_b5 or weight_count == 30389784
        assert model != efficientnet_b6 or weight_count == 43040704
        assert model != efficientnet_b7 or weight_count == 66347960
        assert model != efficientnet_b8 or weight_count == 87413142
        assert model != efficientnet_b0b or weight_count == 5288548
        assert model != efficientnet_b1b or weight_count == 7794184
        assert model != efficientnet_b2b or weight_count == 9109994
        assert model != efficientnet_b3b or weight_count == 12233232
        assert model != efficientnet_b4b or weight_count == 19341616
        assert model != efficientnet_b5b or weight_count == 30389784
        assert model != efficientnet_b6b or weight_count == 43040704
        assert model != efficientnet_b7b or weight_count == 66347960

        x = torch.randn(1, 3, net.in_size[0], net.in_size[1])
        y = net(x)
        y.sum().backward()
        assert tuple(y.size()) == (1, 1000)


@BACKBONES.register_module()
class OTXEfficientNet(EfficientNet):
    """
    Create EfficientNet model with specific parameters.
    Parameters:
    ----------
    version : str
        Version of EfficientNet ('b0'...'b8').
    in_size : tuple of two ints
        Spatial size of the expected input image.
    """

    def __init__(self, version, **kwargs):
        self.model_name = "efficientnet_" + version

        if version == "b0":
            in_size = (224, 224)
            depth_factor = 1.0
            width_factor = 1.0
        elif version == "b1":
            in_size = (240, 240)
            depth_factor = 1.1
            width_factor = 1.0
        elif version == "b2":
            in_size = (260, 260)
            depth_factor = 1.2
            width_factor = 1.1
        elif version == "b3":
            in_size = (300, 300)
            depth_factor = 1.4
            width_factor = 1.2
        elif version == "b4":
            in_size = (380, 380)
            depth_factor = 1.8
            width_factor = 1.4
        elif version == "b5":
            in_size = (456, 456)
            depth_factor = 2.2
            width_factor = 1.6
        elif version == "b6":
            in_size = (528, 528)
            depth_factor = 2.6
            width_factor = 1.8
        elif version == "b7":
            in_size = (600, 600)
            depth_factor = 3.1
            width_factor = 2.0
        elif version == "b8":
            in_size = (672, 672)
            depth_factor = 3.6
            width_factor = 2.2
        else:
            raise ValueError("Unsupported EfficientNet version {}".format(version))

        init_block_channels = 32
        layers = [1, 2, 2, 3, 3, 4, 1]
        downsample = [1, 1, 1, 1, 0, 1, 0]
        channels_per_layers = [16, 24, 40, 80, 112, 192, 320]
        expansion_factors_per_layers = [1, 6, 6, 6, 6, 6, 6]
        kernel_sizes_per_layers = [3, 3, 5, 3, 5, 5, 3]
        strides_per_stage = [1, 2, 2, 2, 1, 2, 1]
        final_block_channels = 1280

        layers = [int(math.ceil(li * depth_factor)) for li in layers]
        channels_per_layers = [
            round_channels(ci * width_factor) for ci in channels_per_layers
        ]

        from functools import reduce

        channels = reduce(
            lambda x, y: x + [[y[0]] * y[1]]
            if y[2] != 0
            else x[:-1] + [x[-1] + [y[0]] * y[1]],
            zip(channels_per_layers, layers, downsample),
            [],
        )
        kernel_sizes = reduce(
            lambda x, y: x + [[y[0]] * y[1]]
            if y[2] != 0
            else x[:-1] + [x[-1] + [y[0]] * y[1]],
            zip(kernel_sizes_per_layers, layers, downsample),
            [],
        )
        expansion_factors = reduce(
            lambda x, y: x + [[y[0]] * y[1]]
            if y[2] != 0
            else x[:-1] + [x[-1] + [y[0]] * y[1]],
            zip(expansion_factors_per_layers, layers, downsample),
            [],
        )
        strides_per_stage = reduce(
            lambda x, y: x + [[y[0]] * y[1]]
            if y[2] != 0
            else x[:-1] + [x[-1] + [y[0]] * y[1]],
            zip(strides_per_stage, layers, downsample),
            [],
        )
        strides_per_stage = [si[0] for si in strides_per_stage]

        init_block_channels = round_channels(init_block_channels * width_factor)

        if width_factor > 1.0:
            assert int(final_block_channels * width_factor) == round_channels(
                final_block_channels * width_factor
            )
            final_block_channels = round_channels(final_block_channels * width_factor)

        super().__init__(
            channels=channels,
            init_block_channels=init_block_channels,
            final_block_channels=final_block_channels,
            kernel_sizes=kernel_sizes,
            strides_per_stage=strides_per_stage,
            expansion_factors=expansion_factors,
            dropout_cls=dict(dist="none"),
            tf_mode=False,
            bn_eps=1e-5,
            in_size=in_size,
            **kwargs,
        )
        self.init_weights(self.pretrained)

    def forward(self, x):
        return super().forward(x, return_featuremaps=True)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str) and os.path.exists(pretrained):
            load_checkpoint(self, pretrained)
            logger.info(f"init weight - {pretrained}")
        elif pretrained is not None:
            load_checkpoint(self, pretrained_urls[self.model_name])
            logger.info(f"init weight - {pretrained_urls[self.model_name]}")
