# NOTE: import order is critical for now: extensions, openvino and only then numpy
from openvino_extensions import get_extensions_path
from openvino.runtime import Core

import subprocess
import pytest
from pathlib import Path

import numpy as np

def convert_model():
    subprocess.run(['mo',
                    '--input_model=model.onnx',
                    # '--extension', "user_ie_extensions/build/libuser_cpu_extension.so"],
                    '--extension', get_extensions_path()],
                    check=True)

def run_test(convert_ir=True, test_onnx=False, num_inputs=1, threshold=1e-5):
    if convert_ir and not test_onnx:
        convert_model()

    inputs = {}
    shapes = {}
    for i in range(num_inputs):
        suffix = '{}'.format(i if i > 0 else '')
        data = np.load('inp' + suffix + '.npy')
        inputs['input' + suffix] = data
        shapes['input' + suffix] = data.shape

    ref = np.load('ref.npy')

    ie = Core()
    # ie.add_extension("user_ie_extensions/build/libuser_cpu_extension.so")
    ie.add_extension(get_extensions_path())
    # ie.set_config({'CONFIG_FILE': 'user_ie_extensions/gpu_extensions.xml'}, 'GPU')

    net = ie.read_model('model.onnx' if test_onnx else 'model.xml')
    net.reshape(shapes)
    exec_net = ie.compile_model(net, 'CPU')

    out = exec_net.infer_new_request(inputs)
    out = next(iter(out.values()))

    assert ref.shape == out.shape
    diff = np.max(np.abs(ref - out))
    assert diff <= threshold


# def test_unpool():
#     from examples.unpool.export_model import export
#     export(mode='default')
#     run_test()


# def test_unpool_reshape():
#     from examples.unpool.export_model import export
#     export(mode='dynamic_size', shape=[5, 3, 6, 9])
#     run_test()

#     export(mode='dynamic_size', shape=[4, 3, 17, 8])
#     run_test(convert_ir=False)

@pytest.mark.parametrize("shape", [[5, 120, 2], [4, 240, 320, 2], [3, 16, 240, 320, 2], [4, 5, 16, 31, 2]])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("centered", [False, True])
@pytest.mark.parametrize("test_onnx", [False, True])
@pytest.mark.parametrize("dims", [[1], [1, 2], [2, 3]])
def test_fft(shape, inverse, centered, test_onnx, dims):
    from examples.fft.export_model import export

    if len(shape) == 3 and dims != [1] or \
       len(shape) == 4 and dims == [2, 3] or \
       len(shape) == 5 and dims == [1] or \
       centered and len(dims) != 2:
        pytest.skip("unsupported configuration")

    export(shape, inverse, centered, dims)
    run_test(test_onnx=test_onnx)


@pytest.mark.parametrize("test_onnx", [False, True])
def test_grid_sample(test_onnx):
    from examples.grid_sample.export_model import export

    export()
    run_test(num_inputs=2, test_onnx=test_onnx)


@pytest.mark.parametrize("shape", [[3, 2, 4, 8, 2], [3, 1, 4, 8, 2]])
@pytest.mark.parametrize("test_onnx", [False, True])
def test_complex_mul(shape, test_onnx):
    from examples.complex_mul.export_model import export

    export(other_shape=shape)
    run_test(num_inputs=2, test_onnx=test_onnx)


@pytest.mark.parametrize("in_channels", [1, 3])
@pytest.mark.parametrize("filters", [1, 4])
@pytest.mark.parametrize("kernel_size", [[3, 3, 3], [5, 5, 5], [2, 2, 2]])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("out_pos", [None, 16])
def test_sparse_conv(in_channels, filters, kernel_size, normalize, out_pos):
    from examples.sparse_conv.export_model import export

    export(num_inp_points=1000, num_out_points=out_pos, max_grid_extent=4, in_channels=in_channels,
           filters=filters, kernel_size=kernel_size, normalize=normalize,
           transpose=False)
    run_test(num_inputs=3, test_onnx=True, threshold=1e-4)


@pytest.mark.parametrize("in_channels", [1, 3])
@pytest.mark.parametrize("filters", [1, 4])
@pytest.mark.parametrize("kernel_size", [[3, 3, 3], [5, 5, 5]])
@pytest.mark.parametrize("normalize", [False])
@pytest.mark.parametrize("out_pos", [None, 16])
def test_sparse_conv_transpose(in_channels, filters, kernel_size, normalize, out_pos):
    from examples.sparse_conv.export_model import export

    export(num_inp_points=1000, num_out_points=out_pos, max_grid_extent=4, in_channels=in_channels,
           filters=filters, kernel_size=kernel_size, normalize=normalize,
           transpose=True)
    run_test(num_inputs=3, test_onnx=True, threshold=1e-4)


def test_calculate_grid():
    from examples.calculate_grid.export_model import export
    export(num_points=10, max_grid_extent=5)
    run_test(test_onnx=True)


def test_deformable_conv():
    from examples.deformable_conv.export_model import export

    export(
        inplanes=15,
        outplanes=15,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        deformable_groups=1,
        inp_shape=[1, 15, 128, 240],
        offset_shape=[1, 18, 128, 240],
    )
    run_test(num_inputs=2, threshold=2e-5)
    run_test(num_inputs=2, test_onnx=True, threshold=2e-5)
