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

def arg_to_list(arg):
    if arg and not isinstance(arg, (list, tuple)):
        arg = (arg,)
    return arg


class BaseHook:
    def __call__(self, context, node, inputs, outputs=None):
        pass


class InsertAfterNodeHook(BaseHook):
    def __init__(self, fn, nodes=None, keys=None):
        if nodes is None:
            self.nodes = keys
        else:
            self.nodes = {(n['type'], tuple(n['scope']), n['context']) for n in nodes}
        self.fn = fn

    def _must_run_callback(self, context, node, inputs, outputs):
        return (node['type'], tuple(node['scope']), node['context']) in self.nodes

    def __call__(self, context, node, inputs, outputs=None):
        if self._must_run_callback(context, node, inputs, outputs):
            return self.fn(context, outputs, node)
        return None
