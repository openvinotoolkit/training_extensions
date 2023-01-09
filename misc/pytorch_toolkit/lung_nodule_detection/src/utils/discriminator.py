import torch
from torch import nn

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

def ch_shuffle(x):
    shuffIdx1 = torch.from_numpy(np.random.randint(0,2,x.size(0)))
    shuffIdx2 = 1-shuffIdx1
    d_in = torch.Tensor(x.size()).cuda()
    d_in[:,shuffIdx1] = x[:,0]
    d_in[:,shuffIdx2] = x[:,1]
    shuffLabel = torch.cat((shuffIdx1.unsqueeze(1),shuffIdx2.unsqueeze(1)),dim=1)
    return d_in, shuffLabel
