"""
 Copyright (c) 2021 Intel Corporation

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

import torch
from utils import get_shape

#class to simplify work with states
class States():
    def __init__(self, state_old, state=None):
        self.state_old =  None if state_old is None else state_old.copy()
        self.state = [] if state is None else state.copy()

    def update(self, state):
        if state is not None:
            self.state += [s.detach() for s in state]
        if self.state_old is not None:
            self.state_old = self.state_old[len(state):]

    def pad_left(self, x, size, dim, shift_right=False):
        if self.state_old is None:
            shape = get_shape(x)
            shape[dim] = size
            x_pad = torch.zeros(shape, dtype=x.dtype, device=x.device)
        else:
            x_pad = self.state_old[0]

        #add left part of x
        x_padded = torch.cat([x_pad, x], dim)

        #get right part for padding on next iter
        x_splited = torch.split(x_padded, [get_shape(x_padded)[dim]-size, size], dim=dim)

        self.update(x_splited[-1:])

        return x_splited[0] if shift_right else x_padded
