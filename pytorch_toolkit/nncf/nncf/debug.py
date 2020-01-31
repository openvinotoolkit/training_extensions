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

import logging
import shutil
import warnings
from pathlib import Path
from typing import List, Dict

import numpy as np
from torch import Tensor
from torch.nn import Module


def is_debug():
    return logging.getLogger().getEffectiveLevel() == logging.DEBUG


logger = logging.getLogger(__name__)


class CallCountTracker:
    def __init__(self, name):
        self.name = name
        self.call_counts = {}

    def init_with_key_list(self, key_list: List):
        self.call_counts = {key: 0 for key in key_list}
        logger.debug("{} tracker: registered {} entries".format(self.name, len(self.call_counts)))

    def register_call(self, key, counts=None):
        if key not in self.call_counts:
            warnings.warn("DEBUG: {} tracker: called an unregistered module: {}".format(self.name, key))
            return
        if counts is None:
            self.call_counts[key] += 1
        else:
            self.call_counts[key] = counts

    def get_never_called_keys(self) -> List[str]:
        return [k for k, v in self.call_counts.items() if v == 0]

    def get_overcalled_keys_with_call_counts(self) -> Dict[str, int]:
        return {k: v for k, v in self.call_counts.items() if v > 1}

    def get_total_call_count(self) -> int:
        if self.call_counts:
            return sum(self.call_counts.values())
        return 0

    def reset(self):
        for key in self.call_counts:
            self.call_counts[key] = 0


class DebugInterface:
    def pre_forward_actions(self, module: Module):
        raise NotImplementedError

    def post_forward_actions(self, module: Module):
        raise NotImplementedError


def debuggable_forward(forward_func):
    def decorated(self, *args, **kwargs):
        if self.debug_interface is not None:
            self.debug_interface.pre_forward_actions(self)
        retval = forward_func(self, *args, **kwargs)
        if self.debug_interface is not None:
            self.debug_interface.post_forward_actions(self)
        return retval

    return decorated


class QuantizationDebugInterface(DebugInterface):
    QUANTIZED_MODULES_TRACKER_NAME = 'quantized_modules'
    ACTIVATION_QUANTIZERS_TRACKER_NAME = 'activation_quantizers'
    FUNCTION_QUANTIZERS_TRACKER_NAME = 'function_quantizers'

    def __init__(self):
        self.call_trackers = {
            self.QUANTIZED_MODULES_TRACKER_NAME: CallCountTracker(
                QuantizationDebugInterface.QUANTIZED_MODULES_TRACKER_NAME),
            self.ACTIVATION_QUANTIZERS_TRACKER_NAME: CallCountTracker(
                QuantizationDebugInterface.ACTIVATION_QUANTIZERS_TRACKER_NAME),
            self.FUNCTION_QUANTIZERS_TRACKER_NAME: CallCountTracker(
                self.FUNCTION_QUANTIZERS_TRACKER_NAME)
        }
        self.graph_size = 0
        self.dump_dir = Path("debug_dumps")
        self.scale_dump_dir = self.dump_dir / Path("scale")
        self.forward_call_count = 0
        self._strict_forward = False

    def init_actual(self, quantizer_module_id_list, activation_quantizer_id_list: List[str],
                    function_input_quantizer_id_list: List[str]):
        self.call_trackers[self.QUANTIZED_MODULES_TRACKER_NAME].init_with_key_list(quantizer_module_id_list)
        self.call_trackers[self.ACTIVATION_QUANTIZERS_TRACKER_NAME].init_with_key_list(activation_quantizer_id_list)
        self.call_trackers[self.FUNCTION_QUANTIZERS_TRACKER_NAME].init_with_key_list(
            function_input_quantizer_id_list)
        if self.scale_dump_dir.exists():
            shutil.rmtree(str(self.scale_dump_dir))
        self.scale_dump_dir.mkdir(parents=True, exist_ok=True)
        self._strict_forward = True

    def pre_forward_actions(self, module: 'QuantizedNetwork'):
        self.reset_counters()

    def post_forward_actions(self, module: 'QuantizedNetwork'):
        self.register_forward_call()
        from nncf.dynamic_graph.context import get_context
        # pylint:disable=protected-access
        ctx = get_context(module._context_name)
        self.set_graph_size(ctx.graph.get_nodes_count())
        for qm_name, qm_module in module.all_quantizations.items():
            # Important - this will not work for DataParallel since it copies the
            # entire parent module for each thread and the `call_count` attributes
            # are incremented for thread local copies of `qm_module`, which are not
            # the same as the master copies of `qm_module` iterated over at this point
            self.register_quantizer_module_call(qm_name, qm_module.call_count)
            self.dump_scale(qm_module.get_trainable_params(), qm_name)
            qm_module.reset_call_counter()
        self.print_call_stats()

        call_dict = ctx.get_node_call_counter_dict()
        total_calls = sum(call_dict.values())
        logger.debug("{} nodes called out of total {}".format(total_calls,
                                                              ctx.graph.get_nodes_count()))
        print()
        if self._strict_forward:
            for tracker in self.call_trackers.values():
                if tracker.get_never_called_keys():
                    # This will always trigger for DataParallel - disregard or disable debug mode
                    # for DataParallel runs
                    raise RuntimeError("{} has never called modules: {}!".format(
                        tracker.name, tracker.get_never_called_keys()))

    def dump_scale(self, quantizer_scale_params: Dict[str, Tensor], quantizer_name: str):
        import re
        quantizer_normalized_name = re.sub(r'[^\w\-_\. ]', '_', quantizer_name)
        for scale_param_name, scale_param in quantizer_scale_params.items():
            fname = "{}_{}.txt".format(quantizer_normalized_name, scale_param_name)
            with open(str(self.scale_dump_dir / fname), "ba") as file:
                np.savetxt(file, scale_param.cpu().numpy())

    def reset_counters(self):
        for tracker in self.call_trackers.values():
            tracker.reset()

    def register_quantizer_module_call(self, key, counts=None):
        self.call_trackers[self.QUANTIZED_MODULES_TRACKER_NAME].register_call(key, counts)

    def register_activation_quantize_call(self, key: str):
        self.call_trackers[self.ACTIVATION_QUANTIZERS_TRACKER_NAME].register_call(key)

    def register_function_quantizer_call(self, key: str):
        self.call_trackers[self.FUNCTION_QUANTIZERS_TRACKER_NAME].register_call(key)

    def print_call_stats(self):
        logger.debug(" Graph size: {} nodes".format(self.graph_size))
        for tracker in self.call_trackers.values():
            msg = " {} tracker:".format(tracker.name)
            msg += " {} total calls;".format(tracker.get_total_call_count())

            never_called = tracker.get_never_called_keys()
            if never_called:
                msg += " {} entries never called;".format(len(never_called))

            overcalled = tracker.get_overcalled_keys_with_call_counts()
            if overcalled:
                msg += " {} entries called more than once;".format(len(overcalled))
            logger.debug(msg)

    def set_graph_size(self, new_size):
        if new_size != self.graph_size:
            logger.debug('\n')
            logger.debug(" warning - graph size has increased from {} to {} since last forward".format(self.graph_size,
                                                                                                       new_size))
        self.graph_size = new_size

    def register_forward_call(self):
        self.forward_call_count += 1
