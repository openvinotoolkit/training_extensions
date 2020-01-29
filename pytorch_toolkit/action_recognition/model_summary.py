from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch import nn

from action_recognition.model import create_model
from action_recognition.models.modules.self_attention import (
    MultiHeadAttention, ScaledDotProductAttention
)
from action_recognition.options import parse_arguments


class LayerStatistic(object):
    def __init__(self):
        self.flops = 0
        self.count = 0


class FlopsCounter(object):
    def __init__(self, types, count_fn):
        self.types = types
        self.count_fn = count_fn


class Layer:
    def __init__(self):
        self.flops = 0
        self.params = 0
        self.module = None
        self.inputs = None
        self.outputs = None
        self.type = None

        self.name = None
        self.out_edges = []


counter_fns = {}


def count_flops_impl(layer_types, key=None):
    if key is None:
        type_ = layer_types[0] if isinstance(layer_types, tuple) else layer_types
        key = type_.__name__

    def decorator(counter):
        counter_fns[key] = FlopsCounter(layer_types, counter)

        def wrap(module, input, output):
            counter(module, input, output)

        return wrap

    return decorator


@count_flops_impl(torch.nn.Conv2d)
def count_flops_conv2d(module, input, output):
    _, _, output_h, output_w = output.size()
    output_c, input_c, kernel_w, kernel_h = module.weight.data.size()

    flops = (kernel_w * kernel_h * input_c + 1) * output_c * output_h * output_w / module.groups

    params = module.weight.numel()
    if module.bias is not None:
        params += module.bias.numel()
    return flops, params


@count_flops_impl(torch.nn.Conv3d)
def count_flops_conv3d(module, input, output):
    _, _, output_t, output_h, output_w = output.size()
    output_c, input_c, kernel_t, kernel_w, kernel_h = module.weight.data.size()

    flops = (kernel_t * kernel_w * kernel_h * input_c + 1) * output_c * output_h * output_w * output_t

    params = module.weight.numel()
    if module.bias is not None:
        params += module.bias.numel()
    return flops, params


@count_flops_impl(torch.nn.BatchNorm2d)
def count_flops_bn2d(module, input, output):
    _, output_c, output_h, output_w = output.size()

    flops = output_c * output_h * output_w

    params = module.weight.numel()
    if module.bias is not None:
        params += module.bias.numel()
    return flops, params


@count_flops_impl(torch.nn.BatchNorm3d)
def count_flops_bn3d(module, input, output):
    _, output_t, output_c, output_h, output_w = output.size()

    flops = output_c * output_h * output_w * output_t

    params = module.weight.numel()
    if module.bias is not None:
        params += module.bias.numel()
    return flops, params


@count_flops_impl(torch.nn.MaxPool2d)
def count_flops_maxpool2d(module, input, output):
    _, output_c, output_h, output_w = output.size()
    kernel_w, kernel_h = module.kernel_size, module.kernel_size

    flops = kernel_w * kernel_h * output_c * output_h * output_w

    return flops, 0


@count_flops_impl(torch.nn.Conv1d)
def count_flops_conv1d(module, input, output):
    _, _, output_w = output.size()
    output_c, input_c, kernel_w = module.weight.data.size()

    flops = (kernel_w * input_c + 1) * output_c * output_w

    params = module.weight.numel()
    if module.bias is not None:
        params += module.bias.numel()
    return flops, params


@count_flops_impl(torch.nn.Linear)
def count_flops_fc(module, inputs, output):
    flops = (module.in_features + 1) * module.out_features * np.prod(inputs[0].shape[:-1])
    params = module.weight.data.numel() + module.bias.data.numel()

    return flops, params


@count_flops_impl(torch.nn.ReLU)
def count_flops_relu(module, input, output):
    flops = np.prod(output.shape[1:])
    return flops, 0


@count_flops_impl(ScaledDotProductAttention)
def count_flops_attention(module, inputs, output):
    q, k, v = inputs
    # does not count softmax
    # 2 x bmm only
    w_flops = q.size(0) * q.size(1) * q.size(2) * k.size(1)
    attend_flops = 2 * q.size(0) * q.size(1) * k.size(1) * v.size(1)
    flops = w_flops + attend_flops
    return flops, 0


@count_flops_impl(MultiHeadAttention)
def count_flops_mh_attention(module, inputs, output):
    q, k, v = inputs

    # does not count inner attention layer
    q_t_flops = module.w_qs.size(0) * module.w_qs.size(1) * q.size(1) * module.n_head
    k_t_flops = module.w_ks.size(0) * module.w_ks.size(1) * k.size(1) * module.n_head
    v_t_flops = module.w_vs.size(0) * module.w_vs.size(1) * v.size(1) * module.n_head

    flops = q_t_flops + k_t_flops + v_t_flops
    params = module.w_qs.numel() + module.w_ks.numel() + module.w_vs.numel()
    return flops, params


@count_flops_impl(torch.nn.LSTM)
def count_flops_LSTM(module, input, output):
    num_layers = module.num_layers
    input_size = module.input_size
    hidden_size = module.hidden_size
    sigmoid_flops = 1
    tanh_flops = 1
    i_t_flops = (hidden_size * input_size) + hidden_size + (hidden_size * hidden_size) + hidden_size + (
            sigmoid_flops * hidden_size)
    f_t_flops = i_t_flops
    o_t_flops = i_t_flops
    g_t_flops = (hidden_size * input_size) + hidden_size + (hidden_size * hidden_size) + hidden_size + (
            tanh_flops * hidden_size)
    c_t_flops = 2 * hidden_size
    h_t_flops = hidden_size * tanh_flops + hidden_size

    flops = i_t_flops + f_t_flops + g_t_flops + o_t_flops + c_t_flops + h_t_flops
    flops *= num_layers

    params = 0
    for k, v in module._parameters.items():
        params += np.prod(v.shape)

    return flops, params


layer_summary = defaultdict(LayerStatistic)

all_layers = []
named_layers = {}

module_to_layer = {}


def compute_flops(module, inputs, outputs):
    for key, counter in counter_fns.items():
        if isinstance(module, counter.types):
            return counter.count_fn(module, inputs, outputs)
    return 0, 0


def compute_layer_statistics_hook(module, input, output):
    new_layer = Layer()
    new_layer.module = module
    new_layer.inputs = input
    new_layer.outputs = output
    flops, params = compute_flops(module, input, output)
    new_layer.flops = flops
    new_layer.params = params

    if isinstance(module, nn.Conv2d):
        new_layer.type = "Conv2d ({})".format("x".join(str(i) for i in module.kernel_size))
    else:
        new_layer.type = type(module).__name__

    if new_layer.type not in ('Identity', 'Dropout', 'BasicBlock', 'Sequential'):
        all_layers.append(new_layer)


def restore_module_names(model: torch.nn.Module):
    for name, module in model.named_modules():
        for layer in all_layers:
            if module is layer.module:
                layer.name = name
                named_layers[name] = layer

                module_to_layer[module] = layer

    # connect edges
    for module, layer in module_to_layer.items():
        for name, children in module.named_children():
            if children not in module_to_layer:
                continue

            if module_to_layer[children].inputs is layer.outputs:
                layer.out_edges.append(module_to_layer[children])


def human_readable_fmt(value):
    if value > 1e9:
        return "{:.2f}G".format((value / 1e9))
    if value > 1e6:
        return "{:.2f}M".format((value / 1e6))
    return "{:.2f}M".format((value / 1e6))


def print_statistics():
    layers_table = []

    for layer in all_layers:
        layers_table.append({
            'name': layer.name,
            'type': layer.type,
            'flops': layer.flops,
            'input_shape': ", ".join(str(tuple(in_.shape)) for in_ in layer.inputs if isinstance(in_, torch.Tensor)),
            'params': layer.params
        })

    data_frame = pd.DataFrame(layers_table)

    total_flops = data_frame.flops.sum()
    total_params = data_frame.params.sum()

    data_frame['flops_'] = data_frame.flops.transform(human_readable_fmt)
    data_frame['params_'] = data_frame.params.transform(human_readable_fmt)
    data_frame['flops%'] = data_frame.flops / total_flops * 100
    data_frame_ = data_frame[['name', 'type', 'flops_', 'flops%', 'params_', 'input_shape']]

    with pd.option_context('display.max_rows', None, 'display.max_columns', 10, 'display.max_colwidth', 60,
                           'display.width', 320, 'display.float_format', lambda f: "{:.2f}".format(f)):
        print(data_frame_)

    print("Total Flops: ", human_readable_fmt(total_flops))
    print("Total params: ", human_readable_fmt(total_params))


def main():
    args = parse_arguments()
    net, _ = create_model(args, args.model)
    net = net.module
    net.cpu()

    if net is None:
        return
    net.eval()

    h, w = args.sample_size, args.sample_size
    var = torch.randn(1, args.sample_duration, 3, h, w).to('cpu')

    net.apply(lambda m: m.register_forward_hook(compute_layer_statistics_hook))

    out = net(var)

    restore_module_names(net)
    print_statistics()


if __name__ == '__main__':
    main()
