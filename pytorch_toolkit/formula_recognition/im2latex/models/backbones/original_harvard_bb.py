import torch.nn as nn

IM2LATEX_BB_LAYERS = ['cnn_encoder']

class Im2LatexBackBone(nn.Module):
    def __init__(self):
        # follow the original paper's table2: CNN specification
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 512, 3, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), (1, 2), (0, 0)),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1), (0, 0)),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0)),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (1, 1))
        )

    def forward(self, imgs):
        return self.cnn_encoder(imgs)
