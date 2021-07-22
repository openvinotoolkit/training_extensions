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

from sc_sdk.entities.label import Color, Label, distinct_colors
from sc_sdk.entities.label_schema import (LabelGroup, LabelGroupType,
                                          LabelSchema)


def generate_label_schema(label_names, label_domain):
    colors = distinct_colors(len(label_names)) if len(label_names) > 0 else []
    not_empty_labels = [Label(name=name, color=colors[i], domain=label_domain, id=i) for i, name in
                        enumerate(label_names)]
    emptylabel = Label(name=f"Empty label", color=Color(42, 43, 46),
                       is_empty=True, domain=label_domain, id=len(not_empty_labels))

    label_schema = LabelSchema()
    exclusive_group = LabelGroup(name="labels", labels=not_empty_labels, group_type=LabelGroupType.EXCLUSIVE)
    empty_group = LabelGroup(name="empty", labels=[emptylabel], group_type=LabelGroupType.EMPTY_LABEL)
    label_schema.add_group(exclusive_group)
    label_schema.add_group(empty_group, exclusive_with=[exclusive_group])
    return label_schema
