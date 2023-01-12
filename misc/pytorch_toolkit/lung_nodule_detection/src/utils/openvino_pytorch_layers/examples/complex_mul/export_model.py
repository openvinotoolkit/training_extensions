import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from .complex_mul import ComplexMul

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.complex_mul = ComplexMul()

    def forward(self, x, y):
        return self.complex_mul.apply(x, y)

def export(inp_shape=[3, 2, 4, 8, 2], other_shape=[3, 2, 4, 8, 2]):
    np.random.seed(324)
    torch.manual_seed(32)

    model = MyModel()
    inp = Variable(torch.randn(inp_shape))
    inp1 = Variable(torch.randn(other_shape))
    model.eval()

    with torch.no_grad():
        torch.onnx.export(model, (inp, inp1), 'model.onnx',
                        input_names=['input', 'input1'],
                        output_names=['output'],
                        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

    ref = model(inp, inp1)
    np.save('inp', inp.detach().numpy())
    np.save('inp1', inp1.detach().numpy())
    np.save('ref', ref.detach().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate ONNX model and test data')
    parser.add_argument('--inp_shape', type=int, nargs='+', default=[3, 2, 4, 8, 2])
    parser.add_argument('--other_shape', type=int, nargs='+', default=[3, 2, 4, 8, 2])
    args = parser.parse_args()

    export(args.inp_shape, args.other_shape)
