import torch.nn as nn
import torchvision.models

architectures = {
    "resnet18": torchvision.models.resnet18,
    "resnet34": torchvision.models.resnet34,
    "resnet50": torchvision.models.resnet50,
    "resnet101": torchvision.models.resnet101,
    "resnet152": torchvision.models.resnet152,
    "resnext50_32x4d": torchvision.models.resnext50_32x4d,
    "resnext101_32x8d": torchvision.models.resnext101_32x8d,
}


class ResNetLikeBackbone(nn.Module):
    def __init__(self, configuration):
        super(ResNetLikeBackbone, self).__init__()
        disable_layer_3 = configuration.get('disable_layer_3')
        disable_layer_4 = configuration.get('disable_layer_4')
        arch = configuration.get('arch')
        in_lstm_ch = configuration.get('in_lstm_ch')
        enable_last_conv = configuration.get('enable_last_conv')
        self.arch = arch
        if in_lstm_ch is not None:
            self.in_lstm_ch = in_lstm_ch
        else:
            self.in_lstm_ch = 64
        self._resnet = architectures.get(arch, None)(
            pretrained=True, progress=True)
        self.groups = self._resnet.groups
        self.base_width = self._resnet.base_width
        self.conv1 = self._resnet.conv1
        self.bn1 = self._resnet.bn1
        self.relu = self._resnet.relu
        self.maxpool = self._resnet.maxpool
        self.layer1 = self._resnet.layer1
        self.layer2 = self._resnet.layer2
        enable_layer_3 = not disable_layer_3
        enable_layer_4 = not disable_layer_4
        if arch == 'resnet18' or arch == 'resnet34':
            in_ch = 128
        else:
            in_ch = 512
        if enable_layer_4:
            assert enable_layer_3, "Cannot enable layer4 w/out enabling layer 3"

        if enable_layer_3 and disable_layer_4:
            self.layer3 = self._resnet.layer3
            self.layer4 = None
            if arch == 'resnet18' or arch == 'resnet34':
                in_ch = 256
            else:
                in_ch = 1024
        elif enable_layer_3 and enable_layer_4:
            self.layer3 = self._resnet.layer3
            self.layer4 = self._resnet.layer4
            if arch == 'resnet18' or arch == 'resnet34':
                in_ch = 512
            else:
                in_ch = 2048
        else:
            self.layer3 = None
            self.layer4 = None
        del self._resnet
        print("Initialized cnn encoder {}".format(arch))
        if enable_last_conv:
            print("Last conv enabled")
            self.last_conv = nn.Conv2d(in_ch, self.in_lstm_ch, 1)
        else:
            self.last_conv = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        if self.layer3 is not None:
            x = self.layer3(x)
        if self.layer4 is not None:
            x = self.layer4(x)
        if self.last_conv is not None:
            x = self.last_conv(x)
        return x
