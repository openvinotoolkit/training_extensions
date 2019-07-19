import torch.nn as nn
# import torch
from .common import ModelInterface


class CNN6(ModelInterface):
    """Facial landmarks localization network"""
    def __init__(self):
        super(CNN6, self).__init__()
        self.bn_first = nn.BatchNorm2d(3)
        activation = nn.ReLU
        self.landnet = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            activation(),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            activation(),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            activation(),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            activation(),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            activation(),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(512)
        )
        self.fc_loc = nn.Conv2d(512, 32, kernel_size=2, padding=0)
        self.init_weights()

    def forward(self, x):
        xs = self.landnet(self.bn_first(x))
        xs = self.fc_loc(xs)
        return xs

    def get_input_res(self):
        return 64, 64

    def set_dropout_ratio(self, ratio):
        pass

    def init_weights(self):
    """Initializes weights of the model before training"""
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.zero_()
# def main():
#     input = torch.randint(0, 255, (2, 3, 64, 64), dtype=torch.float32)

#     model = CNN6()
#     out = model.forward(input)
#     print(out.shape)

# main()