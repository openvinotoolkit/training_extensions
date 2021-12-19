import torch
import torch.nn as nn
import torch.nn.functional as F
from   torchvision import models

# Encoder Part of U-Net architecture 

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(EncoderBlock, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(in_channels, n_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        return x

# Decoder part of U-Net architecture
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, transpose=True):
        super(DecoderBlock, self).__init__()

        if transpose == True:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2)

        self.conv1x1 = nn.Conv2d(in_channels, n_filters, kernel_size=1, padding=0)

        self.conv1 = nn.Conv2d(in_channels, n_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_filters)

        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_filters)

    def forward(self, x_up, x_down):
        x_up = self.up(x_up)
        x_up = self.conv1x1(x_up)

        x = torch.cat([x_up, x_down], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        return x

# Input Block: First Conv Layer Block applied to the input image 

class InBlock(nn.Module):
    def __init__(self, in_channels=1, n_filters=16):
        super(InBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_filters)
        self.conv3 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(n_filters)


    def forward(self, x):


        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        return x


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

# Construct the U-Net Architecture using the Input, Encoder and Decoder part of the network defined above

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_filters=16):
        super(UNet, self).__init__()

        flt = num_filters
        self.layer1 = InBlock(in_channels, flt) # 640x320
        self.layer2 = EncoderBlock(flt, flt*2) # 320x160
        self.layer3 = EncoderBlock(flt*2, flt*2*2) # 160x80
        self.layer4 = EncoderBlock(flt*2*2, flt*2*2*2) # 80x40
        self.layer5 = EncoderBlock(flt*2*2*2, flt*2*2*2*2) # 40x20
        self.layer6 = EncoderBlock(flt*2*2*2*2, flt*2*2*2*2*2) # 20x10
        
        self.layer7 = DecoderBlock(flt*2*2*2*2*2, flt*2*2*2*2)
        self.layer8 = DecoderBlock(flt*2*2*2*2, flt*2*2*2)
        self.layer9 = DecoderBlock(flt*2*2*2,flt*2*2)
        self.layer10 = DecoderBlock(flt*2*2, flt*2)
        self.layer11 = DecoderBlock(flt*2, flt)

        self.out = nn.Conv2d(flt, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)

        x = self.layer7(x6, x5)
        x = self.layer8(x, x4)
        x = self.layer9(x, x3)
        x = self.layer10(x, x2)
        x = self.layer11(x, x1)

        x = self.out(x)

        return x

if __name__ == '__main__':

    from torch.autograd import Variable
    from torchvision import models

    X = Variable(torch.rand(4, 1, 640, 320))

    model = UNet(1, 1)
    print(model)
    model(X)
