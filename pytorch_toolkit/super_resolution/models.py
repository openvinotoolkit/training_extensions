import math
import torch
import torch.nn as nn

def make_model(name_of_the_model, scale):
    if name_of_the_model == 'SRResNetLight':
        model = SRResNetLight(scale=scale).cuda()
    elif name_of_the_model == 'SmallModel':
        model = SmallModel(scale=scale).cuda()
    else:
        print('ERROR:', name_of_the_model, ' model is not supported.')
        raise NotImplementedError

    return model


class ResBlock(nn.Module):
    def __init__(self, num_of_channels):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=num_of_channels, out_channels=num_of_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(num_of_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=num_of_channels, out_channels=num_of_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(num_of_channels, affine=True)

    def forward(self, x):
        orig = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output, orig)
        return output


class SRResNetLight(nn.Module):
    def __init__(self, scale=4, num_of_res_blocks=16, num_of_channels=32):
        super(SRResNetLight, self).__init__()

        self.scale = scale

        self.conv_input = nn.Conv2d(in_channels=3, out_channels=num_of_channels, kernel_size=9, stride=1, padding=4,
                                    bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.residual = self.make_layer(ResBlock(num_of_channels), num_of_res_blocks)

        self.conv_mid = nn.Conv2d(in_channels=num_of_channels, out_channels=num_of_channels, kernel_size=3, stride=1,
                                  padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(num_of_channels, affine=True)

        if scale == 2:
            factor = 2
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=num_of_channels, out_channels=num_of_channels*factor*factor,
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(factor),
                nn.ReLU(inplace=True),
            )
        elif scale == 4:
            factor = 2
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=num_of_channels, out_channels=num_of_channels*factor*factor,
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(factor),
                nn.ReLU(inplace=True),

                nn.Conv2d(num_of_channels, num_of_channels, groups=num_of_channels, kernel_size=3, padding=1, stride=1,
                          bias=False),

                nn.Conv2d(num_of_channels, num_of_channels*factor*factor, kernel_size=1, bias=True),

                nn.PixelShuffle(factor),
                nn.ReLU(inplace=True),
            )
        elif scale == 3:
            factor = 3
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=num_of_channels, out_channels=num_of_channels*factor*factor, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(factor),
                nn.ReLU(inplace=True),
            )
        else:
            raise NotImplementedError

        self.conv_output = nn.Conv2d(in_channels=num_of_channels, out_channels=3, kernel_size=9, stride=1, padding=4,
                                        bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block)
        return nn.Sequential(*layers)

    def forward(self, x):
        input = x
        if isinstance(x, (list, tuple)):
            input = x[0]

        out = self.relu(self.conv_input(input))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out, residual)
        out = self.upscale(out)
        out = self.conv_output(out)
        return [out]


class SmallBlock(nn.Module):
    def __init__(self, channels):
        super(SmallBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1,
                               bias=False)

    def forward(self, x):
        identity_data = x
        output = self.relu(x)
        output = self.conv1(output)
        output = self.relu(output)
        output = self.conv2(output)

        output = torch.add(output, identity_data)
        return output

class SmallModel(nn.Module):
    def __init__(self, scale=3, num_of_ch_enc=16, num_of_ch_dec=8, num_of_res_blocks=4):
        super(SmallModel, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=3, out_channels=num_of_ch_enc, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()

        self.conv_cubic1 = nn.Conv2d(in_channels=3, out_channels=num_of_ch_dec, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_cubic2 = nn.Conv2d(in_channels=num_of_ch_dec, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)

        self.residual1 = SmallBlock(num_of_ch_enc)
        self.residual2 = SmallBlock(num_of_ch_enc)
        self.residual3 = SmallBlock(num_of_ch_enc)
        self.residual4 = SmallBlock(num_of_ch_enc)

        self.conv_mid = nn.Conv2d(in_channels=num_of_ch_enc * (num_of_res_blocks + 1), out_channels=num_of_ch_dec, kernel_size=3, stride=1, padding=1, bias=True)

        if scale == 4:
            factor = 2
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=num_of_ch_dec, out_channels=num_of_ch_dec * factor * factor,
                          kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(factor),
                nn.ReLU(inplace=True),

                nn.Conv2d(num_of_ch_dec, num_of_ch_dec * factor * factor, kernel_size=3, padding=1, stride=1, bias=True),
                nn.PixelShuffle(factor),
                nn.ReLU(inplace=True)
            )
        elif scale == 3:
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=num_of_ch_dec, out_channels=num_of_ch_dec * scale * scale,
                          kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(scale),
                nn.ReLU(inplace=True)
            )
        else:
            raise NotImplementedError

        self.conv_output = nn.Conv2d(in_channels=num_of_ch_dec, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        input = x[0]
        cubic = x[1]

        c1 = self.conv_cubic1(cubic)
        c1 = self.relu(c1)
        c2 = self.conv_cubic2(c1)
        c2 = self.sigmoid(c2)

        in1 = self.conv_input(input)

        out = self.relu(in1)

        out1 = self.residual1(out)
        out2 = self.residual2(out1)
        out3 = self.residual3(out2)
        out4 = self.residual4(out3)

        out = torch.cat([out, out1, out2, out3, out4], dim=1)
        out = self.conv_mid(out)
        out = self.relu(out)
        out = self.upscale(out)
        out = self.conv_output(out)

        return [torch.add(out*c2, cubic)]


class MSE_loss(nn.Module):
    def __init__(self, border=4):
        super(MSE_loss, self).__init__()
        self.border = border

    def forward(self, x, y):
        assert (x[0].shape == y[0].shape)

        h, w = x[0].shape[2:]

        x = x[0][:, :, self.border:h - self.border, self.border:w - self.border]
        y = y[0][:, :, self.border:h - self.border, self.border:w - self.border]

        diff = x - y
        diff = diff ** 2

        return torch.mean(diff)
