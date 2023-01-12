import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from .sparse_conv import SparseConvONNX, SparseConvTransposeONNX


def export(num_inp_points, num_out_points, max_grid_extent, in_channels,
           filters, kernel_size, normalize, transpose):
    np.random.seed(324)
    torch.manual_seed(32)

    if transpose:
        sparse_conv = SparseConvTransposeONNX(in_channels=in_channels,
                                              filters=filters,
                                              kernel_size=kernel_size,
                                              use_bias=False,
                                              normalize=False)
    else:
        sparse_conv = SparseConvONNX(in_channels=in_channels,
                                     filters=filters,
                                     kernel_size=kernel_size,
                                     use_bias=False,
                                     normalize=False)

    # Generate a list of unique positions and add a mantissa
    def gen_pos(num_points):
        inp_pos = np.random.randint(0, max_grid_extent, [num_points, 3])
        inp_pos = np.unique(inp_pos, axis=0).astype(np.float32)
        inp_pos = torch.tensor(inp_pos) + torch.rand(inp_pos.shape, dtype=torch.float32) # [0, 1)
        return inp_pos

    inp_pos = gen_pos(num_inp_points)
    out_pos = gen_pos(num_out_points) if num_out_points else inp_pos

    features = torch.randn([inp_pos.shape[0], in_channels])

    voxel_size = torch.tensor(1.0)
    sparse_conv.eval()

    new_kernel = torch.randn(sparse_conv.state_dict()["kernel"].shape)
    sparse_conv.load_state_dict({"kernel": new_kernel,
                                 "offset": sparse_conv.state_dict()["offset"]})

    with torch.no_grad():
        torch.onnx.export(sparse_conv, (features, inp_pos, out_pos, voxel_size), 'model.onnx',
                          input_names=['input', 'input1', 'input2', 'voxel_size'],
                          output_names=['output'],
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

    ref = sparse_conv(features, inp_pos, out_pos, voxel_size)
    np.save('inp', features.detach().numpy())
    np.save('inp1', inp_pos.detach().numpy())
    np.save('inp2', out_pos.detach().numpy())
    np.save('ref', ref.detach().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate ONNX model and test data')
    parser.add_argument('--num_points', type=int)
    parser.add_argument('--max_grid_extent', type=int)
    parser.add_argument('--in_channels', type=int)
    parser.add_argument('--filters', type=int)
    parser.add_argument('--kernel_size', type=int)
    args = parser.parse_args()

    export(args.num_points, args.max_grid_extent,
           args.in_channels, args.filters, args.kernel_size)
