import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class SpectralConv2d(nn.Module):
    def __init__(self, ch, k, p, py, s, dil):
        # 3x3  =>    ch=, k=3, p=1, s=1
        super().__init__()
        self.ch = ch

        self.ydim = nn.Conv2d(
            ch, ch, (k, 1), padding=(
                py, 0), stride=(
                s, 1), dilation=(
                dil, 1), bias=False)
        self.xdim = nn.Conv2d(
            ch, ch, (1, k), padding=(
                0, p), stride=(
                1, s), groups=ch, bias=False)
        self.bias = nn.Conv2d(
            ch, ch, (1, 1), padding=(
                0, 0), stride=(
                1, 1), groups=ch, bias=True)

    def forward(self, x):
        # do y x first follwed by the other spectal decompsed filters!

        op = self.ydim(x)
        # print(op.shape, x.shape)
        op = self.xdim(op)
        op = self.bias(op)

        return op


class SpectralConv2dInter(nn.Module):
    def __init__(self, ch, k, pad, dil, wts_list=None, a=1):
        super().__init__()
        self.kernel = k + (k - 1) * (dil - 1)  # default dil = 1(no dilation)
        self.pad = pad
        index = torch.Tensor(list(range(0, self.kernel, dil))).to(torch.int64)
        z = torch.zeros(ch, ch, self.kernel, 1)
        if wts_list is None:
            # initialize the wts for training
            print("Not implemented")
        ydim_wt = wts_list.ydim.weight.data

        self.wts_norm = F.pad(
            ydim_wt, (0, 0, (self.kernel - k) // 2, (self.kernel - k) // 2))
        self.wts_dil = z.index_add_(2, index, ydim_wt)

        self.wts = torch.nn.Parameter(
            self.wts_norm * (1 - a) + self.wts_dil * a)

        self.xdim = nn.Conv2d(
            ch, ch, (1, k), padding=(
                0, 1), stride=(
                1, 1), groups=ch, bias=False)
        self.bias = nn.Conv2d(
            ch, ch, (1, 1), padding=(
                0, 0), stride=(
                1, 1), groups=ch, bias=True)

        if wts_list is not None:
            self.xdim.load_state_dict(wts_list.xdim.state_dict())
            self.bias.load_state_dict(wts_list.bias.state_dict())

    def interpolate(self, a):
        self.wts.data = self.wts_norm * a + self.wts_dil * (1 - a)

    def forward(self, x):

        op = F.conv2d(x, self.wts, padding=(self.pad, 0))

        op = self.xdim(op)
        op = self.bias(op)
        return op


class SpectralConv3dInter(nn.Module):
    def __init__(self, ch, k, pad, dil):
        super().__init__()
        self.kernel = k + (k - 1) * (dil - 1)  # default dil = 1(no dilation)
        self.pad = pad
        p = int((pad == 2))

        self.ydim = nn.Conv3d(
            ch, ch, (1, self.kernel, 1), padding=(
                0, pad, 0), stride=(
                1, 1, 1), bias=False)
        self.xdim = nn.Conv3d(
            ch, ch, (1, 1, k), padding=(
                0, 0, p), stride=(
                1, 1, 1), groups=ch, bias=False)
        self.zdim = nn.Conv3d(
            ch, ch, (k, 1, 1), padding=(
                p, 0, 0), stride=(
                1, 1, 1), groups=ch, bias=False)
        self.bias = nn.Conv3d(
            ch, ch, (1, 1, 1), padding=(
                0, 0, 0), stride=(
                1, 1, 1), groups=ch, bias=True)

    def forward(self, x):

        # F.conv3d(x, self.wts_norm, padding=(0, self.pad, 0), stride=1)
        op = self.ydim(x)
        select_x = np.array([np.random.randint(2) for i in range(op.shape[1])])
        select_z = 1 - select_x

        opx = self.xdim(op)
        opz = self.zdim(op)
        for i in range(op.shape[1]):
            op[:, i, :, :] = select_x[i] * opx[:, i, :, :] + \
                select_z[i] * opz[:, i, :, :]

        op = self.bias(op)
        return op


class ResidualBlock(nn.Module):
    def __init__(self, ch, py, dil):
        super().__init__()
        self.conv = nn.Sequential(
            SpectralConv2d(ch=ch, k=3, p=1, py=py, s=1, dil=dil),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(inplace=True),
            SpectralConv2d(ch=ch, k=3, p=1, py=py, s=1, dil=dil),
            nn.BatchNorm2d(ch)
        )
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = out + residual
        out = self.relu(out)
        return out


class ResidualBlockInter(nn.Module):
    def __init__(self, ch, py, dil, wts_list=None, a=1):
        super().__init__()
        self.conv = nn.Sequential(
            SpectralConv2dInter(
                ch=ch,
                k=3,
                pad=py,
                dil=dil,
                wts_list=wts_list.conv[0],
                a=a),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(
                inplace=True),
            SpectralConv2dInter(
                ch=ch,
                k=3,
                pad=py,
                dil=dil,
                wts_list=wts_list.conv[3],
                a=a),
            nn.BatchNorm2d(ch))
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = out + residual
        out = self.relu(out)
        return out


class ResidualBlock3dInter(nn.Module):
    def __init__(self, ch, py, dil):
        super().__init__()
        self.conv = nn.Sequential(
            SpectralConv3dInter(
                ch=ch,
                k=3,
                pad=py,
                dil=dil),
            nn.BatchNorm3d(ch),
            nn.LeakyReLU(
                inplace=True),
            SpectralConv3dInter(
                ch=ch,
                k=3,
                pad=py,
                dil=dil),
            nn.BatchNorm3d(ch))
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = out + residual
        out = self.relu(out)
        return out


class GeneratorModel(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.down1 = nn.Sequential(
            # 1x1 for channel change
            nn.Conv2d(in_ch, 16, 1, padding=0, stride=1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            # 3x3 spectral conv
            SpectralConv2d(16, k=3, p=1, py=1, s=1, dil=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            # 1x1 for channel change
            nn.Conv2d(16, 32, 1, padding=0, stride=1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            # 3x3 spectral conv
            SpectralConv2d(32, k=3, p=1, py=1, s=1, dil=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True)
        )

        self.res1 = ResidualBlock(32, py=1, dil=1)
        self.res2 = ResidualBlock(32, py=1, dil=1)
        self.res3 = ResidualBlock(32, py=1, dil=1)
        self.res4 = ResidualBlock(32, py=1, dil=1)

        self.down2 = nn.Sequential(
            # 1x1 for channel change
            nn.Conv2d(32, 16, 1, padding=0, stride=1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            # 3x3 spectral conv
            SpectralConv2d(16, k=3, p=1, py=1, s=1, dil=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            # 1x1 for channel change
            nn.Conv2d(16, 1, 1, padding=0, stride=1, bias=True),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True),
            # 3x3 spectral conv
            SpectralConv2d(1, k=1, p=0, py=0, s=1, dil=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.down1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.down2(x)

        return x


class GeneratorInter(nn.Module):
    def __init__(self, in_ch, model=None, a=1):
        super().__init__()

        self.down1 = nn.Sequential(
            # 1x1 for channel change
            nn.Conv2d(in_ch, 16, 1, padding=0, stride=1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            # 3x3 spectral conv
            SpectralConv2dInter(
                16, k=3, pad=2, dil=2, wts_list=model.down1[3], a=a),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            # 1x1 for channel change
            nn.Conv2d(16, 32, 1, padding=0, stride=1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            # 3x3 spectral conv
            SpectralConv2dInter(32, k=3, pad=2, dil=2,
                                wts_list=model.down1[-3], a=a),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True)
        )

        self.res1 = ResidualBlockInter(
            32, py=2, dil=2, wts_list=model.res1, a=a)
        self.res2 = ResidualBlockInter(
            32, py=2, dil=2, wts_list=model.res2, a=a)
        self.res3 = ResidualBlockInter(
            32, py=2, dil=2, wts_list=model.res3, a=a)
        self.res4 = ResidualBlockInter(
            32, py=2, dil=2, wts_list=model.res4, a=a)

        self.down2 = nn.Sequential(
            # 1x1 for channel change
            nn.Conv2d(32, 16, 1, padding=0, stride=1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            # 3x3 spectral conv
            SpectralConv2dInter(
                16, k=3, pad=2, dil=2, wts_list=model.down2[3], a=a),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            # 1x1 for channel change
            nn.Conv2d(16, 1, 1, padding=0, stride=1, bias=True),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True),
            # 3x3 spectral conv
            SpectralConv2d(1, k=1, p=0, py=0, s=1, dil=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.down1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.down2(x)

        return x


class Generator3dInter(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.down1 = nn.Sequential(
            # 1x1 for channel change
            nn.Conv3d(in_ch, 16, 1, padding=0, stride=1, bias=True),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(inplace=True),
            # 3x3 spectral conv
            SpectralConv3dInter(
                16, k=3, pad=2, dil=2),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(inplace=True),
            # 1x1 for channel change
            nn.Conv3d(16, 32, 1, padding=0, stride=1, bias=True),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(inplace=True),
            # 3x3 spectral conv
            SpectralConv3dInter(32, k=3, pad=2, dil=2),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(inplace=True)
        )

        self.res1 = ResidualBlock3dInter(32, py=2, dil=2)
        self.res2 = ResidualBlock3dInter(32, py=2, dil=2)
        self.res3 = ResidualBlock3dInter(32, py=2, dil=2)
        self.res4 = ResidualBlock3dInter(32, py=2, dil=2)

        self.down2 = nn.Sequential(
            # 1x1 for channel change
            nn.Conv3d(32, 16, 1, padding=0, stride=1, bias=True),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(inplace=True),
            # 3x3 spectral conv
            SpectralConv3dInter(
                16, k=3, pad=2, dil=2),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(inplace=True),
            # 1x1 for channel change
            nn.Conv3d(16, 1, 1, padding=0, stride=1, bias=True),
            nn.BatchNorm3d(1),
            nn.LeakyReLU(inplace=True),
            # 3x3 spectral conv
            SpectralConv3dInter(1, k=1, pad=0, dil=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.down1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.down2(x)

        return x


class DiscriminatorModel(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 96, 3, padding=1, stride=2),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(96, 64, 3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 32, 3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 1, padding=0, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 2, 1, padding=0, stride=1),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=2 * 8 * 8, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
