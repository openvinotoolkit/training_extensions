"""
 Copyright (c) 2018-2020 Intel Corporation
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

from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn


def save_model_cpu(net, optim, ckpt_fname, epoch, write_solverstate=False):
    """Saves model weights and optimizer state (optionally) to a file"""
    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()
    snapshot_dict = {
        'epoch': epoch,
        'state_dict': state_dict}

    if write_solverstate:
        snapshot_dict['optimizer'] = optim

    torch.save(snapshot_dict, ckpt_fname)


def get_model_parameters_number(model, as_string=True):
    """Returns a total number of trainable parameters in a specified model"""
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not as_string:
        return params_num

    if params_num // 10 ** 6 > 0:
        flops_str = str(round(params_num / 10. ** 6, 2)) + 'M'
    elif params_num // 10 ** 3 > 0:
        flops_str = str(round(params_num / 10. ** 3, 2)) + 'k'
    else:
        flops_str = str(params_num)
    return flops_str


def load_model_state(model, snap, device_id, eval_state=True):
    """Loads model weight from a file produced by save_model_cpu"""
    if device_id != -1:
        location = 'cuda:' + str(device_id)
    else:
        location = 'cpu'
    state_dict = torch.load(snap, map_location=location)['state_dict']

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)

    if device_id != -1:
        model.cuda(device_id)
        cudnn.benchmark = True

    if eval_state:
        model.eval()
    else:
        model.train()

    return model


def flip_tensor(x, dim):
    """Flips a tensor along the specified axis"""
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1, -1, -1),
                                                    ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)
