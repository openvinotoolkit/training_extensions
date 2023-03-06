import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from .unpool import Unpool2d

np.random.seed(324)
torch.manual_seed(32)

class MyModel(nn.Module):
    def __init__(self, mode):
        super(MyModel, self).__init__()
        self.mode = mode
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.conv1 = nn.Conv2d(3, 4, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=1, stride=1)
        self.unpool = Unpool2d()

    def forward(self, x):
        y = self.conv1(x)
        output, indices = self.pool(y)
        conv = self.conv2(output)
        if self.mode == 'default':
            return self.unpool.apply(conv, indices)
        elif self.mode == 'dynamic_size':
            return self.unpool.apply(conv, indices, x)
        else:
            raise Exception('Unknown mode: ' + self.mode)


def export(mode, shape=[5, 3, 6, 8]):
    np.random.seed(324)
    torch.manual_seed(32)

    model = MyModel(mode)
    inp = Variable(torch.randn(shape))
    model.eval()

    with torch.no_grad():
        torch.onnx.export(model, inp, 'model.onnx',
                        input_names=['input'],
                        output_names=['output'],
                        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

    ref = model(inp)
    np.save('inp', inp.detach().numpy())
    np.save('ref', ref.detach().numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ONNX model and test data')
    parser.add_argument('--mode', choices=['default', 'dynamic_size'], help='Specify Unpooling behavior')
    parser.add_argument('--shape', type=int, nargs='+', default=[5, 3, 6, 8])
    args = parser.parse_args()
    export(args.mode, args.shape)
