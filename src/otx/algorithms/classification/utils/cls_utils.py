"""Collection of utils about labels in Classifation Task."""

# Copyright (C) 2022 Intel Corporation
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

# pylint: disable=too-many-nested-blocks, invalid-name

import json
from operator import itemgetter
from typing import Any, Dict

from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.serialization.label_mapper import LabelSchemaMapper


def get_multihead_class_info(label_schema: LabelSchemaEntity):  # pylint: disable=too-many-locals
    """Get multihead info by label schema."""
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
    for i, group in enumerate(exclusive_groups):
        head_idx_to_logits_range[str(i)] = (last_logits_pos, last_logits_pos + len(group))
        last_logits_pos += len(group)
        for j, c in enumerate(group):
            class_to_idx[c] = (i, j)  # group idx and idx inside group
            num_single_label_classes += 1

    # other labels are in multilabel group
    for j, group in enumerate(single_label_groups):
        class_to_idx[group[0]] = (len(exclusive_groups), j)

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
        "empty_multiclass_head_indices": [],
    }
    return mixed_cls_heads_info


def get_cls_inferencer_configuration(label_schema: LabelSchemaEntity):
    """Get classification inferencer config by label schema."""
    multilabel = len(label_schema.get_groups(False)) > 1 and len(label_schema.get_groups(False)) == len(
        label_schema.get_labels(include_empty=False)
    )
    hierarchical = not multilabel and len(label_schema.get_groups(False)) > 1
    multihead_class_info = {}
    if hierarchical:
        multihead_class_info = get_multihead_class_info(label_schema)
    return {
        "multilabel": multilabel,
        "hierarchical": hierarchical,
        "multihead_class_info": multihead_class_info,
        "confidence_threshold": 0.5,
    }


def get_cls_deploy_config(label_schema: LabelSchemaEntity, inference_config: Dict[str, Any]):
    """Get classification deploy config."""
    parameters = {}  # type: Dict[Any, Any]
    parameters["type_of_model"] = "otx_classification"
    parameters["converter_type"] = "CLASSIFICATION"
    parameters["model_parameters"] = inference_config
    parameters["model_parameters"]["labels"] = LabelSchemaMapper.forward(label_schema)
    return parameters


def get_cls_model_api_configuration(label_schema: LabelSchemaEntity, inference_config: Dict[str, Any]):
    """Get ModelAPI config."""
    mapi_config = {}
    mapi_config[("model_info", "model_type")] = "Classification"
    mapi_config[("model_info", "confidence_threshold")] = str(inference_config["confidence_threshold"])
    mapi_config[("model_info", "multilabel")] = str(inference_config["multilabel"])
    mapi_config[("model_info", "hierarchical")] = str(inference_config["hierarchical"])
    mapi_config[("model_info", "output_raw_scores")] = str(True)

    all_labels = ""
    for lbl in label_schema.get_labels(include_empty=False):
        all_labels += lbl.name.replace(" ", "_") + " "
    all_labels = all_labels.strip()
    mapi_config[("model_info", "labels")] = all_labels

    hierarchical_config = {}
    hierarchical_config["cls_heads_info"] = get_multihead_class_info(label_schema)
    hierarchical_config["label_tree_edges"] = []
    for edge in label_schema.label_tree.edges:  # (child, parent)
        hierarchical_config["label_tree_edges"].append((edge[0].name, edge[1].name))

    mapi_config[("model_info", "hierarchical_config")] = json.dumps(hierarchical_config)
    return mapi_config


def get_hierarchical_label_list(hierarchical_info, labels):
    """Return hierarchical labels list which is adjusted to model outputs classes."""
    hierarchical_labels = []
    for head_idx in range(hierarchical_info["num_multiclass_heads"]):
        logits_begin, logits_end = hierarchical_info["head_idx_to_logits_range"][str(head_idx)]
        for logit in range(0, logits_end - logits_begin):
            label_str = hierarchical_info["all_groups"][head_idx][logit]
            label_idx = hierarchical_info["label_to_idx"][label_str]
            hierarchical_labels.append(labels[label_idx])

    if hierarchical_info["num_multilabel_classes"]:
        logits_begin = hierarchical_info["num_single_label_classes"]
        logits_end = len(labels)
        for logit_idx, logit in enumerate(range(0, logits_end - logits_begin)):
            label_str_idx = hierarchical_info["num_multiclass_heads"] + logit_idx
            label_str = hierarchical_info["all_groups"][label_str_idx][0]
            label_idx = hierarchical_info["label_to_idx"][label_str]
            hierarchical_labels.append(labels[label_idx])
    return hierarchical_labels
