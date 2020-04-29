"""
 Copyright (c) 2020 Intel Corporation
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
import pytest

from nncf.dynamic_graph.context import Scope
from nncf.pruning.filter_pruning.algo import FilterPruningBuilder
from nncf.pruning.utils import get_rounded_pruned_element_number, get_bn_for_module_scope, \
    get_first_pruned_modules, get_last_pruned_modules
from tests.pruning.test_helpers import get_basic_pruning_config, BigPruningTestModel, \
    PruningTestModelBranching
from tests.test_helpers import create_compressed_model_and_algo_for_test


# pylint: disable=protected-access
@pytest.mark.parametrize("total,sparsity_rate,multiple_of,ref",
                         [(20, 0.2, None, 4),
                          (20, 0.2, 8, 4),
                          (20, 0.1, 2, 2),
                          (20, 0.1, 5, 0),
                          (20, 0.5, None, 4)
                          ])
def test_get_rounded_pruned_element_number(total, sparsity_rate, multiple_of, ref):
    if multiple_of is not None:
        result = get_rounded_pruned_element_number(total, sparsity_rate, multiple_of)
    else:
        result = get_rounded_pruned_element_number(total, sparsity_rate)
    assert ref == result

    if multiple_of is not None:
        assert (total - result) % multiple_of == 0


def test_get_bn_for_module_scope():
    config = get_basic_pruning_config(input_sample_size=(1, 1, 8, 8))
    config['compression']['algorithm'] = 'filter_pruning'
    pruned_model, _ = create_compressed_model_and_algo_for_test(BigPruningTestModel(), config)

    conv1_scope = Scope.from_str('BigPruningTestModel/NNCFConv2d[conv1]')
    bn = get_bn_for_module_scope(pruned_model, conv1_scope)
    assert bn is None

    conv2_scope = Scope.from_str('BigPruningTestModel/NNCFConv2d[conv2]')
    bn = get_bn_for_module_scope(pruned_model, conv2_scope)
    assert bn == pruned_model.bn

    conv3_scope = Scope.from_str('BigPruningTestModel/NNCFConv2d[conv3]')
    bn = get_bn_for_module_scope(pruned_model, conv3_scope)
    assert bn is None


@pytest.mark.parametrize(('model', 'ref_first_module_names'),
                         [(BigPruningTestModel, ['conv1']),
                          (PruningTestModelBranching, ['conv1', 'conv2', 'conv3']),
                          ],
                         )
def test_get_first_pruned_layers(model, ref_first_module_names):
    config = get_basic_pruning_config(input_sample_size=(1, 1, 8, 8))
    config['compression']['algorithm'] = 'filter_pruning'
    pruned_model, _ = create_compressed_model_and_algo_for_test(model(), config)

    first_pruned_modules = get_first_pruned_modules(pruned_model,
                                                    FilterPruningBuilder(config).get_types_of_pruned_modules())
    ref_first_modules = [getattr(pruned_model, module_name) for module_name in ref_first_module_names]
    assert set(first_pruned_modules) == set(ref_first_modules)


@pytest.mark.parametrize(('model', 'ref_last_module_names'),
                         [(BigPruningTestModel, ['conv3']),
                          (PruningTestModelBranching, ['conv4', 'conv5']
                           ),
                          ],
                         )
def test_get_last_pruned_layers(model, ref_last_module_names):
    config = get_basic_pruning_config(input_sample_size=(1, 1, 8, 8))
    config['compression']['algorithm'] = 'filter_pruning'
    pruned_model, _ = create_compressed_model_and_algo_for_test(model(), config)

    first_pruned_modules = get_last_pruned_modules(pruned_model,
                                                   FilterPruningBuilder(config).get_types_of_pruned_modules())
    ref_last_modules = [getattr(pruned_model, module_name) for module_name in ref_last_module_names]
    assert set(first_pruned_modules) == set(ref_last_modules)
