import torch
import torch.nn as nn
import math

# from .common import ModelInterface


class DsmNet(nn.Module):
    """Facial landmarks localization network"""
    def __init__(self):
        super(DsmNet, self).__init__()
        self.bn_first = nn.BatchNorm2d(3)
        activation = nn.ReLU
        softmax = nn.Softmax2d

        # feature extract
        self.conv0 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(16, affine=False)
        self.relu0 = activation()
        self.pool0 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16, affine=False)
        self.relu1 = activation()
        self.pool1 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32, affine=False)
        self.relu2 = activation()
        self.pool2 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64, affine=False)
        self.relu3 = activation()
        self.pool3 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128, affine=False)
        self.relu4 = activation()
        self.pool4 = nn.Conv2d(128, 64, kernel_size=3, stride=2)
        self.conv5 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256, affine=False)
        self.relu5 = activation()
        self.conv_fm = nn.Conv2d(256, 512, kernel_size=3)

        # leye
        self.leye = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256, affine=False),
            activation(),
            nn.Dropout2d(p=0.1)
        )

        # heads
        self.leye37x = nn.Sequential(
            nn.Conv2d(256, 60, kernel_size=1),
            softmax()
        )
        self.leye37y = nn.Sequential(
            nn.Conv2d(256, 60, kernel_size=1),
            softmax()
        )
        self.leye38x = nn.Sequential(
            nn.Conv2d(256, 60, kernel_size=1),
            softmax()
        )
        self.leye38y = nn.Sequential(
            nn.Conv2d(256, 60, kernel_size=1),
            softmax()
        )
        self.leye39x = nn.Sequential(
            nn.Conv2d(256, 60, kernel_size=1),
            softmax()
        )
        self.leye39y = nn.Sequential(
            nn.Conv2d(256, 60, kernel_size=1),
            softmax()
        )
        self.leye40x = nn.Sequential(
            nn.Conv2d(256, 60, kernel_size=1),
            softmax()
        )
        self.leye40y = nn.Sequential(
            nn.Conv2d(256, 60, kernel_size=1),
            softmax()
        )
        self.leye41x = nn.Sequential(
            nn.Conv2d(256, 60, kernel_size=1),
            softmax()
        )
        self.leye41y = nn.Sequential(
            nn.Conv2d(256, 60, kernel_size=1),
            softmax()
        )
        self.leye42x = nn.Sequential(
            nn.Conv2d(256, 60, kernel_size=1),
            softmax()
        )
        self.leye42y = nn.Sequential(
            nn.Conv2d(256, 60, kernel_size=1),
            softmax()
        )

        # reye
        self.reye = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256, affine=False),
            activation(),
            nn.Dropout2d(p=0.1)
        )

        # heads
        self.reye43x = nn.Sequential(
            nn.Conv2d(256, 60, kernel_size=1),
            softmax()
        )
        self.reye43y = nn.Sequential(
            nn.Conv2d(256, 60, kernel_size=1),
            softmax()
        )
        self.reye44x = nn.Sequential(
            nn.Conv2d(256, 60, kernel_size=1),
            softmax()
        )
        self.reye44y = nn.Sequential(
            nn.Conv2d(256, 60, kernel_size=1),
            softmax()
        )
        self.reye45x = nn.Sequential(
            nn.Conv2d(256, 60, kernel_size=1),
            softmax()
        )
        self.reye45y = nn.Sequential(
            nn.Conv2d(256, 60, kernel_size=1),
            softmax()
        )
        self.reye46x = nn.Sequential(
            nn.Conv2d(256, 60, kernel_size=1),
            softmax()
        )
        self.reye46y = nn.Sequential(
            nn.Conv2d(256, 60, kernel_size=1),
            softmax()
        )
        self.reye47x = nn.Sequential(
            nn.Conv2d(256, 60, kernel_size=1),
            softmax()
        )
        self.reye47y = nn.Sequential(
            nn.Conv2d(256, 60, kernel_size=1),
            softmax()
        )
        self.reye48x = nn.Sequential(
            nn.Conv2d(256, 60, kernel_size=1),
            softmax()
        )
        self.reye48y = nn.Sequential(
            nn.Conv2d(256, 60, kernel_size=1),
            softmax()
        )


        # mouth
        self.mouth = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256, affine=False),
            activation(),
            nn.Dropout2d(p=0.1)
        )

        # heads
        self.mouth49x = nn.Sequential(
            nn.Conv2d(256, 60, kernel_size=1),
            softmax()
        )
        self.mouth49y = nn.Sequential(
            nn.Conv2d(256, 60, kernel_size=1),
            softmax()
        )
        self.mouth52x = nn.Sequential(
            nn.Conv2d(256, 60, kernel_size=1),
            softmax()
        )
        self.mouth52y = nn.Sequential(
            nn.Conv2d(256, 60, kernel_size=1),
            softmax()
        )
        self.mouth55x = nn.Sequential(
            nn.Conv2d(256, 60, kernel_size=1),
            softmax()
        )
        self.mouth55y = nn.Sequential(
            nn.Conv2d(256, 60, kernel_size=1),
            softmax()
        )
        self.mouth67x = nn.Sequential(
            nn.Conv2d(256, 60, kernel_size=1),
            softmax()
        )
        self.mouth67y = nn.Sequential(
            nn.Conv2d(256, 60, kernel_size=1),
            softmax()
        )

        # predict
        self.fc_loc = nn.Sequential(
            # nn.Dropout2d(p=0.3),
            nn.Conv2d(1920, 256, kernel_size=1),
            activation(),
            nn.BatchNorm2d(256, affine=False),
            nn.Conv2d(256, 128, kernel_size=1),
            activation(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 32, kernel_size=1)

        )
        self.init_weights()

    def forward(self, x):
        out = self.bn_first(x)
        out = self.conv0(out)
        out = self.bn0(out)
        out = self.relu0(out)
        out = self.pool0(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.pool3(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu4(out)
        out = self.pool4(out)
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu5(out)
        feature = self.conv_fm(out)

        leye_f = self.leye(feature)
        reye_f = self.reye(feature)
        mouth_f = self.mouth(feature)

        lm37x = self.leye37x(leye_f)
        lm37y = self.leye37y(leye_f)
        lm38x = self.leye38x(leye_f)
        lm38y = self.leye38y(leye_f)
        lm39x = self.leye39x(leye_f)
        lm39y = self.leye39y(leye_f)
        lm40x = self.leye40x(leye_f)
        lm40y = self.leye40y(leye_f)
        lm41x = self.leye41x(leye_f)
        lm41y = self.leye41y(leye_f)
        lm42x = self.leye42x(leye_f)
        lm42y = self.leye42y(leye_f)
        lm43x = self.reye43x(reye_f)
        lm43y = self.reye43y(reye_f)
        lm44x = self.reye44x(reye_f)
        lm44y = self.reye44y(reye_f)
        lm45x = self.reye45x(reye_f)
        lm45y = self.reye45y(reye_f)
        lm46x = self.reye46x(reye_f)
        lm46y = self.reye46y(reye_f) 
        lm47x = self.reye47x(reye_f)
        lm47y = self.reye47y(reye_f)
        lm48x = self.reye48x(reye_f)
        lm48y = self.reye48y(reye_f)
        lm49x = self.mouth49x(mouth_f)
        lm49y = self.mouth49y(mouth_f)
        lm52x = self.mouth52x(mouth_f)
        lm52y = self.mouth52y(mouth_f)
        lm55x = self.mouth55x(mouth_f)
        lm55y = self.mouth55y(mouth_f)
        lm67x = self.mouth67x(mouth_f)
        lm67y = self.mouth67y(mouth_f)

        feature = torch.cat((lm37x, lm37y, lm38x, lm38y, lm39x, lm39y, lm40x, lm40y, lm41x, lm41y, lm42x, lm42y,
                             lm43x, lm43y, lm44x, lm44y, lm45x, lm45y, lm46x, lm46y, lm47x, lm47y, lm48x, lm48y,
                             lm49x, lm49y, lm52x, lm52y, lm55x, lm55y, lm67x, lm67y), dim=1)

        out = self.fc_loc(feature)
        # out = torch.flatten(out)
        return out


    def get_input_res(self):
        return 60, 60

    @classmethod
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


def main():
    input = torch.randint(0, 255, (2, 3, 112, 112), dtype=torch.float32)
    print(input)
    model = DsmNet()
    out = model.forward(input)
    print(out.shape)

main()
