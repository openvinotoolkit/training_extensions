import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from .grid_sample import GridSample


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.grid_sample = GridSample()

    def forward(self, x, grid):
        return self.grid_sample.apply(x, grid)


def export(inp_shape=[5, 3, 6, 9], grid_shape=[5, 6, 9, 2]):
    np.random.seed(324)
    torch.manual_seed(32)

    if inp_shape[2] != grid_shape[1]:
        raise Exception('Input height (got {}) should be equal to grid height (got {})'.format(inp_shape[2], grid_shape[1]))
    if inp_shape[3] != grid_shape[2]:
        raise Exception('Input width (got {}) should be equal to grid width (got {})'.format(inp_shape[3], grid_shape[2]))

    model = MyModel()
    inp = Variable(torch.randn(inp_shape))
    grid = torch.Tensor(np.random.uniform(low=-2, high=2, size=grid_shape))
    model.eval()

    with torch.no_grad():
        torch.onnx.export(model, (inp, grid), 'model.onnx',
                        input_names=['input', 'input1'],
                        output_names=['output'],
                        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

    ref = model(inp, grid)
    np.save('inp', inp.detach().numpy())
    np.save('inp1', grid.detach().numpy())
    np.save('ref', ref.detach().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate ONNX model and test data')
    parser.add_argument('--inp_shape', type=int, nargs='+', default=[5, 3, 6, 9])
    parser.add_argument('--grid_shape', type=int, nargs='+', default=[5, 6, 9, 2])
    args = parser.parse_args()

    export(args.inp_shape, args.grid_shape)
