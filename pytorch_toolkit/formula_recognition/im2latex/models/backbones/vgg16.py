import torch.nn as nn


class VGG16Backbone(nn.Module):
    """ VGG16-like backbone. """

    def __init__(self, backbone_dropout=0.0):
        super().__init__()
        self.backbone_dropout = backbone_dropout

        self.cnn_encoder = nn.Sequential(
            # bn0
            nn.BatchNorm2d(momentum=0.1, num_features=1),
            # dropout1, conv1, bn1, pool1
            nn.Dropout(self.backbone_dropout, inplace=True),
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(momentum=0.1, num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # dropout2, conv2, bn2, pool2
            nn.Dropout(p=self.backbone_dropout, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(momentum=0.1, num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # dropout3, conv3, bn3
            nn.Dropout(p=self.backbone_dropout, inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(momentum=0.1, num_features=256),
            nn.ReLU(inplace=True),
            # dropout4, conv4, bn4, pool4
            nn.Dropout(p=self.backbone_dropout, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(momentum=0.1, num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            # dropout5, conv5, bn5
            nn.Dropout(p=self.backbone_dropout, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(momentum=0.1, num_features=512),
            nn.ReLU(inplace=True),
            # dropout6, conv6, bn6, pool6
            nn.Dropout(p=self.backbone_dropout, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(momentum=0.1, num_features=512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            # dropout7, conv7, bn7
            nn.Dropout(p=self.backbone_dropout, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, stride=(2, 1), padding=(0, 1)),
            nn.BatchNorm2d(momentum=0.1, num_features=512),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputdata):
        return self.cnn_encoder(inputdata)
