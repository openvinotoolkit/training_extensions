# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

# pylint: disable=too-many-nested-blocks

import importlib
from operator import itemgetter
from typing import List

from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.label_schema import LabelGroup, LabelGroupType, LabelSchemaEntity
from otx.api.utils.argument_checks import check_input_parameters_type


@check_input_parameters_type()
def generate_label_schema(not_empty_labels: List[LabelEntity], multilabel: bool = False):
    assert len(not_empty_labels) > 1

    label_schema = LabelSchemaEntity()
    if multilabel:
        emptylabel = LabelEntity(name="Empty label", is_empty=True, domain=Domain.CLASSIFICATION)
        empty_group = LabelGroup(name="empty", labels=[emptylabel], group_type=LabelGroupType.EMPTY_LABEL)
        for label in not_empty_labels:
            label_schema.add_group(LabelGroup(name=label.name, labels=[label], group_type=LabelGroupType.EXCLUSIVE))
        label_schema.add_group(empty_group)
    else:
        main_group = LabelGroup(name="labels", labels=not_empty_labels, group_type=LabelGroupType.EXCLUSIVE)
        label_schema.add_group(main_group)
    return label_schema


@check_input_parameters_type()
def get_multihead_class_info(label_schema: LabelSchemaEntity):
    all_groups = label_schema.get_groups(include_empty=False)
    all_groups_str = []
    for g in all_groups:
        group_labels_str = [lbl.name for lbl in g.labels]
        all_groups_str.append(group_labels_str)

    single_label_groups = [g for g in all_groups_str if len(g) == 1]
    exclusive_groups = [sorted(g) for g in all_groups_str if len(g) > 1]
    single_label_groups.sort(key=itemgetter(0))
    exclusive_groups.sort(key=itemgetter(0))
    class_to_idx = {}
    head_idx_to_logits_range = {}
    num_single_label_classes = 0
    last_logits_pos = 0
    for i, g in enumerate(exclusive_groups):
        head_idx_to_logits_range[i] = (last_logits_pos, last_logits_pos + len(g))
        last_logits_pos += len(g)
        for j, c in enumerate(g):
            class_to_idx[c] = (i, j)  # group idx and idx inside group
            num_single_label_classes += 1

    # other labels are in multilabel group
    for j, g in enumerate(single_label_groups):
        class_to_idx[g[0]] = (len(exclusive_groups), j)

    all_labels = label_schema.get_labels(include_empty=False)
    label_to_idx = {lbl.name: i for i, lbl in enumerate(all_labels)}

    mixed_cls_heads_info = {
        "num_multiclass_heads": len(exclusive_groups),
        "num_multilabel_classes": len(single_label_groups),
        "head_idx_to_logits_range": head_idx_to_logits_range,
        "num_single_label_classes": num_single_label_classes,
        "class_to_group_idx": class_to_idx,
        "all_groups": exclusive_groups + single_label_groups,
        "label_to_idx": label_to_idx,
    }
    return mixed_cls_heads_info


@check_input_parameters_type()
def get_task_class(path: str):
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
