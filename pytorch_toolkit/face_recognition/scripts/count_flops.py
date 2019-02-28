"""
 Copyright (c) 2018 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import argparse

import torch
import numpy as np

from model.common import models_backbones, models_landmarks
from utils.utils import get_model_parameters_number


def add_flops_counting_methods(net_main_module):
    """Adds flops counting hooks to the specified module"""
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_flops_count = start_flops_count.__get__(net_main_module) # pylint: disable=E1111, E1120
    net_main_module.stop_flops_count = stop_flops_count.__get__(net_main_module) # pylint: disable=E1111, E1120
    net_main_module.reset_flops_count = reset_flops_count.__get__(net_main_module) # pylint: disable=E1111, E1120
    net_main_module.compute_average_flops_cost = compute_average_flops_cost.__get__(net_main_module) # pylint: disable=E1111, E1120

    net_main_module.reset_flops_count()

    # Adding variables necessary for masked flops computation
    net_main_module.apply(add_flops_mask_variable_or_reset)

    return net_main_module


def compute_average_flops_cost(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Returns current mean flops consumption per image.

    """

    batches_count = self.__batch_counter__
    flops_sum = 0
    for module in self.modules():
        if is_supported_instance(module):
            flops_sum += module.__flops__

    return flops_sum / batches_count


def start_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Activates the computation of mean flops consumption per image.
    Call it before you run the network.

    """
    add_batch_counter_hook_function(self)
    self.apply(add_flops_counter_hook_function)


def stop_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Stops computing the mean flops consumption per image.
    Call whenever you want to pause the computation.

    """
    remove_batch_counter_hook_function(self)
    self.apply(remove_flops_counter_hook_function)


def reset_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Resets statistics computed so far.

    """
    add_batch_counter_variables_or_reset(self)
    self.apply(add_flops_counter_variable_or_reset)


# ---- Internal functions
def is_supported_instance(module):
    """Internal auxiliary function"""
    if isinstance(module, (torch.nn.Conv2d, torch.nn.ReLU, torch.nn.PReLU, torch.nn.ELU, \
                           torch.nn.LeakyReLU, torch.nn.ReLU6, torch.nn.Linear, torch.nn.MaxPool2d, \
                           torch.nn.AvgPool2d, torch.nn.BatchNorm2d)):
        return True

    return False


def empty_flops_counter_hook(module, _, __):
    """Internal auxiliary function"""
    module.__flops__ += 0


def relu_flops_counter_hook(module, input_, _):
    """Counts flops in activations"""
    input_ = input_[0]
    batch_size = input_.shape[0]
    active_elements_count = batch_size
    for val in input_.shape[1:]:
        active_elements_count *= val

    module.__flops__ += active_elements_count


def linear_flops_counter_hook(module, input_, output):
    """Counts flops in linear layers"""
    input_ = input_[0]
    batch_size = input_.shape[0]
    module.__flops__ += batch_size * input_.shape[1] * output.shape[1]


def pool_flops_counter_hook(module, input_, _):
    """Counts flops in max ind avg pooling layers"""
    input_ = input_[0]
    module.__flops__ += np.prod(input_.shape)


def bn_flops_counter_hook(module, input_, _):
    """Counts flops in batch normalization layers"""
    input_ = input_[0]

    batch_flops = np.prod(input_.shape)
    if module.affine:
        batch_flops *= 2
    module.__flops__ += batch_flops


def conv_flops_counter_hook(conv_module, input_, output):
    """Counts flops in convolution layers"""
    # Can have multiple inputs, getting the first one
    input_ = input_[0]

    batch_size = input_.shape[0]
    output_height, output_width = output.shape[2:]

    kernel_height, kernel_width = conv_module.kernel_size
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = kernel_height * kernel_width * in_channels * filters_per_channel

    active_elements_count = batch_size * output_height * output_width

    if conv_module.__mask__ is not None:
        # (b, 1, h, w)
        flops_mask = conv_module.__mask__.expand(batch_size, 1, output_height, output_width)
        active_elements_count = flops_mask.sum()

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0

    if conv_module.bias is not None:
        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    conv_module.__flops__ += overall_flops


def batch_counter_hook(module, input_, _):
    """Internal auxiliary function"""
    # Can have multiple inputs, getting the first one
    input_ = input_[0]
    batch_size = input_.shape[0]
    module.__batch_counter__ += batch_size


def add_batch_counter_variables_or_reset(module):
    """Internal auxiliary function"""
    module.__batch_counter__ = 0


def add_batch_counter_hook_function(module):
    """Internal auxiliary function"""
    if hasattr(module, '__batch_counter_handle__'):
        return

    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle


def remove_batch_counter_hook_function(module):
    """Internal auxiliary function"""
    if hasattr(module, '__batch_counter_handle__'):
        module.__batch_counter_handle__.remove()
        del module.__batch_counter_handle__


def add_flops_counter_variable_or_reset(module):
    """Internal auxiliary function"""
    if is_supported_instance(module):
        module.__flops__ = 0


def add_flops_counter_hook_function(module):
    """Internal auxiliary function"""
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            return

        if isinstance(module, torch.nn.Conv2d):
            handle = module.register_forward_hook(conv_flops_counter_hook)
        elif isinstance(module, (torch.nn.ReLU, torch.nn.PReLU, torch.nn.ELU,
                                 torch.nn.LeakyReLU, torch.nn.ReLU6, torch.nn.Sigmoid)):
            handle = module.register_forward_hook(relu_flops_counter_hook)
        elif isinstance(module, torch.nn.Linear):
            handle = module.register_forward_hook(linear_flops_counter_hook)
        elif isinstance(module, (torch.nn.AvgPool2d, torch.nn.MaxPool2d)):
            handle = module.register_forward_hook(pool_flops_counter_hook)
        elif isinstance(module, torch.nn.BatchNorm2d):
            handle = module.register_forward_hook(bn_flops_counter_hook)
        else:
            handle = module.register_forward_hook(empty_flops_counter_hook)
        module.__flops_handle__ = handle


def remove_flops_counter_hook_function(module):
    """Internal auxiliary function"""
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            module.__flops_handle__.remove()
            del module.__flops_handle__


def add_flops_mask_variable_or_reset(module):
    """Internal auxiliary function"""
    if is_supported_instance(module):
        module.__mask__ = None


def flops_to_string(flops):
    """Converts flops count to a human-readable form"""
    flops_str = ''
    if flops // 10 ** 9 > 0:
        flops_str = str(round(flops / 10. ** 9, 2)) + 'GMac'
    elif flops // 10 ** 6 > 0:
        flops_str = str(round(flops / 10. ** 6, 2)) + 'MMac'
    elif flops // 10 ** 3 > 0:
        flops_str = str(round(flops / 10. ** 3, 2)) + 'KMac'
    else:
        flops_str = str(flops) + 'Mac'
    return flops_str


def main():
    """Runs flops counter"""
    parser = argparse.ArgumentParser(description='Evaluation script for Face Recognition in PyTorch')
    parser.add_argument('--embed_size', type=int, default=128, help='Size of the face embedding.')
    parser.add_argument('--device', type=int, default=0, help='Device to store the model.')
    parser.add_argument('--model', choices=list(models_backbones.keys()) + list(models_landmarks.keys()), type=str,
                        default='rmnet')
    args = parser.parse_args()

    with torch.cuda.device(args.device), torch.no_grad():
        bs = 1
        if args.model in models_landmarks.keys():
            model = add_flops_counting_methods(models_landmarks[args.model]())
            batch = torch.Tensor(bs, 3, *model.get_input_res())
        else:
            net = models_backbones[args.model](embedding_size=args.embed_size, feature=True)
            batch = torch.Tensor(bs, 3, *net.get_input_res())
            model = add_flops_counting_methods(net)

        model.cuda().eval().start_flops_count()
        output = model(batch.cuda())

        print(model)
        print('Output shape: {}'.format(list(output.shape)))
        print('Flops:  {}'.format(flops_to_string(model.compute_average_flops_cost())))
        print('Params: ' + get_model_parameters_number(model))

if __name__ == '__main__':
    main()
