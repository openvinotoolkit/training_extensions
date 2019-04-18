"""
 Copyright (c) 2019 Intel Corporation

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

import gc
import operator as op
import time
from collections import defaultdict
from functools import reduce, wraps

import numpy as np
import torch


class Timer(object):

    def __init__(self, warmup=0, smoothing=0.5, cuda_sync=True):
        self.warmup = warmup
        self.cuda_sync = cuda_sync
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.smoothed_time = 0.
        self.smoothing_alpha = smoothing
        self.min_time = float('inf')
        self.max_time = 0.
        self.reset()

    def tic(self):
        if self.cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.time()

    def toc(self, average=True, smoothed=False):
        if self.cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.diff = time.time() - self.start_time
        self.calls += 1
        if self.calls <= self.warmup:
            return self.diff

        self.total_time += self.diff
        self.average_time = self.total_time / (self.calls - self.warmup)
        self.smoothed_time = self.smoothed_time * self.smoothing_alpha + self.diff * (1.0 - self.smoothing_alpha)
        self.min_time = min(self.min_time, self.diff)
        self.max_time = max(self.max_time, self.diff)
        if average:
            return self.average_time
        elif smoothed:
            return self.smoothed_time
        else:
            return self.diff

    def __enter__(self):
        self.tic()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.toc()

    def reset(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.


class DummyTimer(Timer):
    def __init__(self):
        super().__init__()

    def tic(self):
        pass

    def toc(self, *args, **kwargs):
        return 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def timed(func):
    @wraps(func)
    def wrapper_timer(self, *args, **kwargs):
        if not hasattr(self, '_timers'):
            self._timers = defaultdict(Timer)
        with self._timers[func.__name__]:
            value = func(self, *args, **kwargs)
        return value
    return wrapper_timer


def print_timing_stats(timers, key='average_time'):
    print('{:>40}: {:>10} [{:>10}, {:>10}] {:>10} {:>10}'.format('name', 'average', 'min', 'max', '#calls', 'total'))
    for k, v in sorted(timers.items(), key=lambda x: op.attrgetter(key)(x[1]), reverse=True):
        print('{:>40}: {:10.2f} [{:10.2f}, {:10.2f}] {:10d} {:10.2f}'.format(k, 1000 * v.average_time,
                                                                             1000 * v.min_time, 1000 * v.max_time,
                                                                             v.calls, 1000 * v.total_time))
    print('-' * 40)


def pretty_shape(shape):
    if shape is None:
        return 'None'
    return '×'.join(map(str, shape))


def pretty_size(size, units='G', precision=2, base=1024):
    if units is None:
        if size // (base ** 3) > 0:
            val = str(round(size / (base ** 3), precision))
            units = 'G'
        elif size // (base ** 2) > 0:
            val = str(round(size / (base ** 2), precision))
            units = 'M'
        elif size // base > 0:
            val = str(round(size / base, precision))
            units = 'K'
        else:
            val = str(size)
            units = ''
    else:
        if units == 'G':
            val = str(round(size / (base ** 3), precision))
        elif units == 'M':
            val = str(round(size / (base ** 2), precision))
        elif units == 'K':
            val = str(round(size / base, precision))
        else:
            val = str(size)
    
    return val, units


def dump_tensors(gpu_only=True):
    """Prints a list of the Tensors being tracked by the garbage collector."""
    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if not gpu_only or obj.is_cuda:
                    print('%s:%s%s %s' % (type(obj).__name__,
                                          ' GPU' if obj.is_cuda else '',
                                          ' pinned' if obj.is_pinned else '',
                                          pretty_shape(obj.size())))
                    total_size += obj.numel()
            elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
                if not gpu_only or obj.is_cuda:
                    print('%s → %s:%s%s%s%s %s' % (type(obj).__name__,
                                                   type(obj.data).__name__,
                                                   ' GPU' if obj.is_cuda else '',
                                                   ' pinned' if obj.data.is_pinned else '',
                                                   ' grad' if obj.requires_grad else '',
                                                   ' volatile' if obj.volatile else '',
                                                   pretty_shape(obj.data.size())))
                    total_size += obj.data.numel()
        except Exception as e:
            pass
    print('Total size:', total_size)


def list_allocated_tensors():
    memtable = []
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            memtable.append(dict(obj=obj,
                                 size=(reduce(op.mul, obj.size())
                                       if len(obj.size()) > 0
                                       else 0) * obj.element_size()))
    memtable = sorted(memtable, key=op.itemgetter('size'))
    for i, item in enumerate(memtable):
        obj = item['obj']
        print('{:03}: {:>10} {:>30} {:>25} {:>10}'.format(i, item['size'], str(np.array(obj.shape)),
                                                          str(obj.type()), str(obj.device)))


def list_parameters(module):
    memtable = []
    for name, x in module.named_parameters():
        memtable.append(dict(name=name,
                             shape=np.array(x.data.shape),
                             size=int(x.data.numel() * x.data.element_size()),
                             has_grad=x.requires_grad,
                             grad_shape=np.array(x.grad.shape) if x.requires_grad else None,
                             grad_size=int(x.grad.numel() * x.grad.element_size()) if x.requires_grad else 0
                             )
                        )
    total_data_size = 0
    total_grad_size = 0
    for i, item in enumerate(memtable):
        print('{:03} {:>60}: {:>15} {:>15} {:>15} {:>15}'.format(i,
                                                                 item['name'],
                                                                 pretty_size(item['size'], units='M')[0],
                                                                 pretty_shape(item['shape']),
                                                                 pretty_size(item['grad_size'], units='M')[0],
                                                                 pretty_shape(item['grad_shape'])))
        total_data_size += item['size']
        total_grad_size += item['grad_size']

    total_mem_size = list(pretty_size(total_data_size)) + list(pretty_size(total_grad_size))
    print('TOTAL MEMORY USAGE FOR MODEL PARAMETERS: data: {} {}B grad: {} {}B'.format(*total_mem_size))


class FeatureMapsTracer(object):
    fwd_tensors_registry = set()
    bwd_tensors_registry = set()

    @staticmethod
    def reset(*args, **kwargs):
        FeatureMapsTracer.summary_fwd()
        FeatureMapsTracer.summary_bwd()
        del FeatureMapsTracer.fwd_tensors_registry
        FeatureMapsTracer.fwd_tensors_registry = set()
        del FeatureMapsTracer.bwd_tensors_registry
        FeatureMapsTracer.bwd_tensors_registry = set()

    @staticmethod
    def summary_fwd(*args, **kwargs):
        total_data_size = FeatureMapsTracer.get_total_size(list(FeatureMapsTracer.fwd_tensors_registry))
        print('TOTAL FORWARD DATA BLOBS SIZE: {} {}B'.format(*pretty_size(total_data_size)))

    @staticmethod
    def summary_bwd(*args, **kwargs):
        total_data_size = FeatureMapsTracer.get_total_size(list(FeatureMapsTracer.bwd_tensors_registry))
        print('TOTAL BACKWARD GRAD BLOBS SIZE: {} {}B'.format(*pretty_size(total_data_size)))

    @staticmethod
    def list_tensors(x):
        tensors = []
        if isinstance(x, (list, tuple)):
            for i in x:
                tensors.extend(FeatureMapsTracer.list_tensors(i))
        elif isinstance(x, dict):
            for i in x.values():
                tensors.extend(FeatureMapsTracer.list_tensors(i))
        elif isinstance(x, torch.Tensor):
            tensors.append(x)
        return tensors

    @staticmethod
    def get_shapes(tensors):
        shapes = [x.shape for x in tensors]
        return shapes

    @staticmethod
    def shapes_to_str(shapes):
        return '[' + ', '.join([pretty_shape(shape) for shape in shapes]) + ']'

    @staticmethod
    def get_total_size(tensors):
        total_size = 0
        for x in tensors:
            total_size += int(x.numel() * x.element_size())
        return total_size

    @staticmethod
    def forward(module, inputs, outputs, verbose=False):
        input_tensors = FeatureMapsTracer.list_tensors(inputs)
        inputs_shapes = FeatureMapsTracer.get_shapes(input_tensors)
        inputs_shapes_str = FeatureMapsTracer.shapes_to_str(inputs_shapes)
        inputs_size = FeatureMapsTracer.get_total_size(input_tensors)
        FeatureMapsTracer.fwd_tensors_registry.update(set(input_tensors))

        output_tensors = FeatureMapsTracer.list_tensors(outputs)
        outputs_shapes = FeatureMapsTracer.get_shapes(output_tensors)
        outputs_shapes_str = FeatureMapsTracer.shapes_to_str(outputs_shapes)
        outputs_size = FeatureMapsTracer.get_total_size(output_tensors)
        FeatureMapsTracer.fwd_tensors_registry.update(set(output_tensors))

        if verbose:
            print('fwd {:>20}: {:>15} {:>15} {:>15} {:>15}'.format(module._get_name(),
                                                                   pretty_size(inputs_size, units='M')[0],
                                                                   inputs_shapes_str,
                                                                   pretty_size(outputs_size, units='M')[0],
                                                                   outputs_shapes_str))

    @staticmethod
    def backward(module, inputs, outputs, verbose=False):
        input_tensors = FeatureMapsTracer.list_tensors(inputs)
        inputs_shapes = FeatureMapsTracer.get_shapes(input_tensors)
        inputs_shapes_str = FeatureMapsTracer.shapes_to_str(inputs_shapes)
        inputs_size = FeatureMapsTracer.get_total_size(input_tensors)
        FeatureMapsTracer.bwd_tensors_registry.update(set(input_tensors))

        output_tensors = FeatureMapsTracer.list_tensors(outputs)
        outputs_shapes = FeatureMapsTracer.get_shapes(output_tensors)
        outputs_shapes_str = FeatureMapsTracer.shapes_to_str(outputs_shapes)
        outputs_size = FeatureMapsTracer.get_total_size(output_tensors)
        FeatureMapsTracer.bwd_tensors_registry.update(set(output_tensors))

        if verbose:
            print('bwd {:>20}: {:>15} {:>15} {:>15} {:>15}'.format(module._get_name(),
                                                                   pretty_size(inputs_size, units='M')[0],
                                                                   inputs_shapes_str,
                                                                   pretty_size(outputs_size, units='M')[0],
                                                                   outputs_shapes_str))

    @staticmethod
    def add_fwd_hooks(module):

        def register_per_layer_hooks(m):
            m.register_forward_hook(FeatureMapsTracer.forward)

        module.register_forward_pre_hook(FeatureMapsTracer.reset)
        module.apply(register_per_layer_hooks)

    @staticmethod
    def add_bwd_hooks(module):

        def register_per_layer_hooks(m):
            m.register_backward_hook(FeatureMapsTracer.backward)

        module.apply(register_per_layer_hooks)

    @staticmethod
    def add_hooks(module):
        FeatureMapsTracer.add_fwd_hooks(module)
        FeatureMapsTracer.add_bwd_hooks(module)


class PerformanceCounters(object):
    def __init__(self):
        self.pc = {}

    def update(self, pc):
        for layer, stats in pc.items():
            if layer not in self.pc:
                self.pc[layer] = dict(layer_type=stats['layer_type'],
                                      exec_type=stats['exec_type'],
                                      status=stats['status'],
                                      real_time=stats['real_time'],
                                      calls=1)
            else:
                self.pc[layer]['real_time'] += stats['real_time']
                self.pc[layer]['calls'] += 1

    def print(self):
        print('Performance counters:')
        print(' '.join(['name', 'layer_type', 'exec_type', 'status', 'real_time(us)']))
        for layer, stats in self.pc.items():
            print('{} {} {} {} {}'.format(layer, stats['layer_type'], stats['exec_type'],
                                          stats['status'], stats['real_time'] / stats['calls']))
