import torch.nn as nn

class ConvNetBackbone(nn.Module):
    def __init__(self, input_channels=1, output_channels=80):
        super(ConvNetBackbone, self).__init__()
        num_filters = [16, 32, 48, 64, output_channels]
        self.conv0 = nn.Conv2d(input_channels, num_filters[0], 3, 1, 1)
        self.conv1 = nn.Conv2d(num_filters[0], num_filters[1], 3, 1, 1)
        self.conv2 = nn.Conv2d(num_filters[1], num_filters[2], 3, 1, 1)
        self.conv3 = nn.Conv2d(num_filters[2], num_filters[3], 3, 1, 1)
        self.conv4 = nn.Conv2d(num_filters[3], num_filters[4], 3, 1, 1)

        self.bn0 = nn.BatchNorm2d(num_filters[0])
        self.bn1 = nn.BatchNorm2d(num_filters[1])
        self.bn2 = nn.BatchNorm2d(num_filters[2])
        self.bn3 = nn.BatchNorm2d(num_filters[3])
        self.bn4 = nn.BatchNorm2d(num_filters[4])

        self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)


    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.dropout(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.dropout(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        return x

