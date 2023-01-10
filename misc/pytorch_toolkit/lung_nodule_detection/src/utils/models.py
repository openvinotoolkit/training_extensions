import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from .max_unpool_2d import Unpool2d as MaxUnpool2d


class SUMNet(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()

        self.encoder   = models.vgg11_bn(pretrained = True).features
        self.preconv   = nn.Conv2d(in_ch, 3, 1)
        self.conv1     = self.encoder[0]
        self.bn1       = self.encoder[1]
        self.pool1     = nn.MaxPool2d(2, 2, return_indices = True)
        self.conv2     = self.encoder[4]
        self.bn2       = self.encoder[5]
        self.pool2     = nn.MaxPool2d(2, 2, return_indices = True)
        self.conv3a    = self.encoder[8]
        self.bn3       = self.encoder[9]
        self.conv3b    = self.encoder[11]
        self.bn4       = self.encoder[12]
        self.pool3     = nn.MaxPool2d(2, 2, return_indices = True)
        self.conv4a    = self.encoder[15]
        self.bn5       = self.encoder[16]
        self.conv4b    = self.encoder[18]
        self.bn6       = self.encoder[19]
        self.pool4     = nn.MaxPool2d(2, 2, return_indices = True)
        self.conv5a    = self.encoder[22]
        self.bn7       = self.encoder[23]
        self.conv5b    = self.encoder[25]
        self.bn8       = self.encoder[26]
        self.pool5     = nn.MaxPool2d(2, 2, return_indices = True)

        self.unpool5   = MaxUnpool2d()
        self.donv5b    = nn.Conv2d(1024, 512, 3, padding = 1)
        self.donv5a    = nn.Conv2d(512, 512, 3, padding = 1)
        self.unpool4   = MaxUnpool2d()
        self.donv4b    = nn.Conv2d(1024, 512, 3, padding = 1)
        self.donv4a    = nn.Conv2d(512, 256, 3, padding = 1)
        self.unpool3   = MaxUnpool2d()
        self.donv3b    = nn.Conv2d(512, 256, 3, padding = 1)
        self.donv3a    = nn.Conv2d(256,128, 3, padding = 1)
        self.unpool2   = MaxUnpool2d()
        self.donv2     = nn.Conv2d(256, 64, 3, padding = 1)
        self.unpool1   = MaxUnpool2d()
        self.donv1     = nn.Conv2d(128, 32, 3, padding = 1)
        self.output    = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        preconv        = F.relu(self.preconv(x), inplace = True)
        conv1          = F.relu(self.bn1(self.conv1(preconv)), inplace = True)
        pool1, idxs1   = self.pool1(conv1)
        conv2          = F.relu(self.bn2(self.conv2(pool1)), inplace = True)
        pool2, idxs2   = self.pool2(conv2)
        conv3a         = F.relu(self.bn3(self.conv3a(pool2)), inplace = True)
        conv3b         = F.relu(self.bn4(self.conv3b(conv3a)), inplace = True)
        pool3, idxs3   = self.pool3(conv3b)
        conv4a         = F.relu(self.bn5(self.conv4a(pool3)), inplace = True)
        conv4b         = F.relu(self.bn6(self.conv4b(conv4a)), inplace = True)
        pool4, idxs4   = self.pool4(conv4b)
        conv5a         = F.relu(self.bn7(self.conv5a(pool4)), inplace = True)
        conv5b         = F.relu(self.bn8(self.conv5b(conv5a)), inplace = True)
        pool5, idxs5   = self.pool5(conv5b)

        unpool5        = torch.cat([self.unpool5.apply(pool5, idxs5), conv5b], 1)
        donv5b         = F.relu(self.donv5b(unpool5), inplace = True)
        donv5a         = F.relu(self.donv5a(donv5b), inplace = True)
        unpool4        = torch.cat([self.unpool4.apply(donv5a, idxs4), conv4b], 1)
        donv4b         = F.relu(self.donv4b(unpool4), inplace = True)
        donv4a         = F.relu(self.donv4a(donv4b), inplace = True)
        unpool3        = torch.cat([self.unpool3.apply(donv4a, idxs3), conv3b], 1)
        donv3b         = F.relu(self.donv3b(unpool3), inplace = True)
        donv3a         = F.relu(self.donv3a(donv3b))
        unpool2        = torch.cat([self.unpool2.apply(donv3a, idxs2), conv2], 1)
        donv2          = F.relu(self.donv2(unpool2), inplace = True)
        unpool1        = torch.cat([self.unpool1.apply(donv2, idxs1), conv1], 1)
        donv1          = F.relu(self.donv1(unpool1), inplace = True)
        output         = self.output(donv1)
        return output

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=5)
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3_drop(self.conv3(x)))
        x = self.pool3(x)
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return torch.sigmoid(x)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super().__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)

            x1 = self.conv(x+x1)
        return x1

class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super().__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1

class U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super().__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class R2U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1,t=2):
        super().__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)

        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)

        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)

        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)


        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)

        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)

        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class Discriminator(nn.Module):
    def __init__(self,in_ch, out_ch):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(in_ch, 64, 3, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        nn.MaxPool2d(3),
            # state size. (64) x 32 x 32
            nn.Conv2d(64, 64 * 2, 3, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
        nn.MaxPool2d(3),
            nn.Conv2d(64*2, 64 * 2, 3, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*2) x 16 x 16
            nn.Conv2d(64 * 2, 64 * 4, 3, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
        nn.MaxPool2d(3),
            nn.Conv2d(64 * 4, 64 * 4, 3, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
        nn.MaxPool2d(2),
            # state size. (64*4) x 8 x 8
            nn.Conv2d(64 * 4, out_ch, 7, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Sigmoid()
        )
    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 2) #.squeeze(1)
