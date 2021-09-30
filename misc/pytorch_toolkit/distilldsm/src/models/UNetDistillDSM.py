import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p, bias=True):
        # 3x3  =>    ch=, k=3, p=1, s=1
        super(SpectralConv2d, self).__init__()
        self.bias = bias
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.ydim = nn.Conv2d(in_ch, out_ch, (k,1), padding=(p,0), stride = (s,1), bias = False) 
        self.xdim = nn.Conv2d(out_ch, out_ch, (1,k), padding=(0,p), stride = (1,s), groups=out_ch, bias = False)
        if self.bias:
            self.bias = nn.Conv2d(out_ch, out_ch, (1,1), padding=(0,0), stride = (1,1), groups= out_ch, bias = True) 
        
    def forward(self, x):
        #to_stack = []
        # do y x first follwed by the other spectal decompsed filters! 
        #for i in range(self.out_ch):
        op = self.ydim(x)
        op = self.xdim(op)
        if self.bias:
            op = self.bias(op)
        #to_stack.append(op)
        #x = torch.cat(to_stack, 1) #concatenate along channel dim
        return op

class Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p, bias=True):
        super(Conv2d, self).__init__()
        self.conv_2d = nn.Conv2d(in_ch, out_ch, kernel_size=k,stride=s,padding=p,bias=bias)
    def forward(self,x):
        out = self.conv_2d(x)
        return out

class InplaceShift(torch.autograd.Function):
    # Special thanks to @raoyongming for the help to this function
    @staticmethod
    def forward(ctx, tensor):
        # not support higher order gradient
        # tensor = tensor.detach_()
        t, c, h, w = tensor.size()
        n = 1
        tensor = tensor.view(1, t,c,h,w)
        fold = c // 4
        ctx.fold_ = fold
        buffer_ = tensor.data.new(n, t, fold, h, w).zero_()
        buffer_[:, :-1] = tensor.data[:, 1:, :fold]
        tensor.data[:, :, :fold] = buffer_
        buffer_.zero_()
        buffer_[:, 1:] = tensor.data[:, :-1, fold: 2 * fold]
        tensor.data[:, :, fold: 2 * fold] = buffer_
        return tensor.view(t,c,h,w)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = grad_output.detach_()
        fold = ctx.fold_
        t, c, h, w = grad_output.size()
        n = 1
        grad_output = grad_output.view(1, t,c,h,w)
        buffer_ = grad_output.data.new(n, t, fold, h, w).zero_()
        buffer_[:, 1:] = grad_output.data[:, :-1, :fold]
        grad_output.data[:, :, :fold] = buffer_
        buffer_.zero_()
        buffer_[:, :-1] = grad_output.data[:, 1:, fold: 2 * fold]
        grad_output.data[:, :, fold: 2 * fold] = buffer_
        return grad_output.view(t,c,h,w), None

class learnTSM(nn.Module):
    def __init__(self, in_channels, version='zero', inplace=True):
        super(learnTSM, self).__init__()
        self.split_size = in_channels//2
        self.main_conv = Conv2d(self.split_size*2, self.split_size, k=3, s=1, p=1)
        self.pre_conv = Conv2d(self.split_size*2, self.split_size//2, k=3, s=1, p=1)
        self.post_conv = Conv2d(self.split_size*2, self.split_size//2, k=3, s=1, p=1)
        self.version = version
        self.inplace = inplace

    def forward(self, tensor, tsm_length=16):
        shape = T, C, H, W = tensor.shape
        split_size = self.split_size

        shift_tensor, main_tensor = tensor.split([split_size*2, C - 2 * split_size], dim=1)
        # pre_tensor, post_tensor = shift_tensor.split([split_size, split_size], dim=1)
        pre_tensor = shift_tensor
        post_tensor = shift_tensor
        # print(pre_tensor.shape, post_tensor.shape, shift_tensor.shape, main_tensor.shape, shape)
        main_conv_tensor = self.main_conv(shift_tensor).view(T//tsm_length, tsm_length, split_size, H, W)
        pre_tensor = self.pre_conv(pre_tensor).view(T//tsm_length, tsm_length, split_size//2, H, W)
        post_tensor = self.post_conv(post_tensor).view(T//tsm_length, tsm_length, split_size//2, H, W)
        main_tensor = main_tensor.view(T//tsm_length, tsm_length, C - 2*split_size, H, W)

        if self.version == 'zero':
            pre_tensor  = F.pad(pre_tensor,  (0, 0, 0, 0, 0, 0, 1, 0))[:,  :-1, ...]  # NOQA
            post_tensor = F.pad(post_tensor, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:  , ...]  # NOQA
        elif self.version == 'circulant':
            pre_conv_tensor  = torch.cat((pre_conv_tensor [:, -1:  , ...],  # NOQA
                                     pre_conv_tensor [:,   :-1, ...]), dim=1)  # NOQA
            post_conv_tensor = torch.cat((post_conv_tensor[:,  1:  , ...],  # NOQA
                                     post_conv_tensor[:,   :1 , ...]), dim=1)  # NOQA
        # print(pre_tensor.shape, post_tensor.shape, main_conv_tensor.shape, main_tensor.shape, shape)
        return torch.cat((pre_tensor, post_tensor, main_conv_tensor, main_tensor), dim=2).view(shape)

def tsm_module(tensor, version='zero', inplace=True, tsm_length=16):
    if not inplace:
        shape = T, C, H, W = tensor.shape
        tensor = tensor.view(T//tsm_length, tsm_length, C, H, W)
        split_size = C // 4
        pre_tensor, post_tensor, peri_tensor = tensor.split(
            [split_size, split_size, C - 2 * split_size],
            dim=2
        )
        if version == 'zero':
            pre_tensor  = F.pad(pre_tensor,  (0, 0, 0, 0, 0, 0, 1, 0))[:,  :-1, ...]  # NOQA
            post_tensor = F.pad(post_tensor, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:  , ...]  # NOQA
        elif version == 'circulant':
            pre_tensor  = torch.cat((pre_tensor [:, -1:  , ...],  # NOQA
                                     pre_tensor [:,   :-1, ...]), dim=1)  # NOQA
            post_tensor = torch.cat((post_tensor[:,  1:  , ...],  # NOQA
                                     post_tensor[:,   :1 , ...]), dim=1)  # NOQA
        else:
            raise ValueError('Unknown TSM version: {}'.format(version))
        return torch.cat((pre_tensor, post_tensor, peri_tensor), dim=2).view(shape)
    else:
        out = InplaceShift.apply(tensor)
        return out

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out, conv_type='conv_2d', tsm=False, learn=False, tsm_length=16):
        super(conv_block,self).__init__()
        mod = Conv2d
        if(conv_type == 'spectral'):
            mod = SpectralConv2d
        if(conv_type == 'conv_2d'):
            mod = Conv2d
        self.conv1 = nn.Sequential(
            mod(ch_in, ch_out, k=3,s=1,p=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
            )
        self.conv2 = nn.Sequential(
            mod(ch_out, ch_out, k=3,s=1,p=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        
        self.tsm = tsm
        self.learn = learn
        self.tsm_length = tsm_length
        if self.tsm:
            if self.learn:
                self.tsmConv = learnTSM(in_channels = ch_out)

    def forward(self,x):
        x = self.conv1(x)
        if self.tsm:
            if self.learn:
                x = self.tsmConv(x, tsm_length=self.tsm_length)
            else:
                x = tsm_module(x, 'zero', tsm_length=self.tsm_length).contiguous()
        x = self.conv2(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out, conv_type='conv_2d', tsm = False, learn=False):
        super(up_conv,self).__init__()
        mod = Conv2d
        if(conv_type == 'spectral'):
            mod = SpectralConv2d
        if(conv_type == 'conv_2d'):
            mod = Conv2d
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.co = nn.Sequential(
            mod(ch_in,ch_out,k=3,s=1,p=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )
        self.tsm = tsm
        self.learn = learn
        if self.tsm:
            if self.learn:
                self.tsmConv = learnTSM(in_channels = ch_in)

    def forward(self,x):
        x = self.up(x)
        if self.tsm:
            if self.learn:
                x = self.tsmConv(x)
            else:
                x = tsm_module(x, 'zero').contiguous()
        x = self.co(x)
        return x


class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, ch_in, ch_out, conv_type='conv_2d', tsm = False, learn=False, tsm_length=16):
        super().__init__()

        if(conv_type == 'spectral'):
            mod = SpectralConv2d
        if(conv_type == 'conv_2d'):
            mod = Conv2d
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Sequential(
            mod(ch_in,ch_in//2,k=3,s=1,p=1,bias=True),
            nn.BatchNorm2d(ch_in//2),
            nn.ReLU(inplace=True)
            )
        self.conv2 = nn.Sequential(
            mod(ch_in//2, ch_out, k=3,s=1,p=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.tsm = tsm
        self.learn = learn
        self.tsm_length = tsm_length
        if self.tsm:
            if self.learn:
                self.tsmConv = learnTSM(in_channels = ch_in//2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        x = self.conv1(x)
        if self.tsm:
            if self.learn:
                x = self.tsmConv(x, tsm_length=self.tsm_length)
            else:
                x = tsm_module(x, 'zero', tsm_length=self.tsm_length).contiguous()
        x = self.conv2(x)
        return x


class U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1, conv_type='conv_2d', tsm=False, learn=False, tsm_length = 16):
        super(U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.tsm_length = tsm_length
        self.inp = img_ch
        self.out = output_ch
        self.inc = conv_block(ch_in=img_ch, ch_out=16, conv_type=conv_type)

        self.Conv1 = conv_block(ch_in=16,ch_out=32, conv_type=conv_type, tsm=tsm, learn=learn, tsm_length=tsm_length)
        self.Conv2 = conv_block(ch_in=32,ch_out=64, conv_type=conv_type, tsm=tsm, learn=learn, tsm_length=tsm_length)
        self.Conv3 = conv_block(ch_in=64,ch_out=128, conv_type=conv_type, tsm=tsm, learn=learn, tsm_length=tsm_length)
        self.Conv4 = conv_block(ch_in=128,ch_out=128, conv_type=conv_type, tsm=tsm, learn=learn, tsm_length=tsm_length)

        self.Up1 = Up(ch_in=256,ch_out=64, conv_type=conv_type, tsm=tsm, learn=learn, tsm_length=tsm_length)
        self.Up2 = Up(ch_in=128,ch_out=32, conv_type=conv_type, tsm=tsm, learn=learn, tsm_length=tsm_length)
        self.Up3 = Up(ch_in=64,ch_out=16, conv_type=conv_type, tsm=tsm, learn=learn, tsm_length=tsm_length)        
        self.Up4 = Up(ch_in=32,ch_out=16, conv_type=conv_type, tsm=tsm, learn=learn, tsm_length=tsm_length)

        self.Conv_1x1 = nn.Conv2d(16,output_ch,kernel_size=1,stride=1,padding=0)
        # self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        shape = B, self.inp, T, H, W = x.shape
        # print(x.shape)
        x = x.permute(0, 2, 1, 3, 4)
        # x = x.view(B*T, self.inp, H, W)
        x = x.reshape(B*T, self.inp, H, W)
        # print(x.shape)
        # encoding path
        x1 = self.inc(x)
        # print(x1.shape)
        x2 = self.Maxpool(x1)
        x2 = self.Conv1(x2)
        # print(x2.shape)
        x3 = self.Maxpool(x2)
        x3 = self.Conv2(x3)
        # print(x3.shape)
        x4 = self.Maxpool(x3)
        x4 = self.Conv3(x4)
        # print(x4.shape)
        x5 = self.Maxpool(x4)
        x5 = self.Conv4(x5)
        # print(x5.shape)
        # decoding + concat path
        x = self.Up1(x5, x4)
        # print(x.shape)
        x = self.Up2(x, x3)
        # print(x.shape)
        x = self.Up3(x, x2)
        # print(x.shape)
        x = self.Up4(x, x1)
        # print(x.shape)
        x = self.Conv_1x1(x)
        # print(x.shape)
        # x = self.sigmoid(x)
        x = x.view(B, T, self.out, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        # print(x.shape)
        return x


# net = U_Net(1, 2, conv_type='conv_2d', tsm=True, learn=True)
# model_parameter = filter(lambda p:p.requires_grad, net.parameters())
# params = sum([np.prod(p.size()) for p in model_parameter])
# # print(net)
# print("Total number of trainable params are: ", params)
# print(net(torch.Tensor(32,1, 128, 128)).size())
