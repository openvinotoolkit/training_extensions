"""
Model templates registry.
"""

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

import copy
import glob
import os

import yaml
from ote_sdk.entities.model_template import parse_model_template


class Registry:
    """
    Class that implements a model templates registry.
    """

    def __init__(self, templates_dir=None, templates=None):
        if templates is None:
            if templates_dir is None:
                templates_dir = os.getenv("TEMPLATES_DIR")

            if templates_dir is None:
                raise RuntimeError("The templates_dir is not set.")

            template_filenames = glob.glob(
                os.path.join(templates_dir, "**", "template.yaml"), recursive=True
            )
            template_filenames = [os.path.abspath(p) for p in template_filenames]

            self.templates = []

            for template_file in template_filenames:
                self.templates.append(parse_model_template(template_file))
        else:
            self.templates = copy.deepcopy(templates)

        self.task_types = self.__collect_task_types(self.templates)

    @staticmethod
    def __collect_task_types(templates):
        return {template.task_type for template in templates}

    def filter(self, framework=None, task_type=None):
        """
        Filters registry by framework and/or task type and returns filtered registry.
        """

        templates = copy.deepcopy(self.templates)
        if framework is not None:
            templates = [
                template
                for template in templates
                if template.framework.lower() == framework.lower()
            ]
        if task_type is not None:
            templates = [
                template
                for template in templates
                if str(template.task_type).lower() == task_type.lower()
            ]
        return Registry(templates=templates)

    def get(self, template_id):
        """
        Returns a model template with specified template_id.
        """

        templates = [
            template
            for template in self.templates
            if template.model_template_id == template_id
        ]
        if not templates:
            raise ValueError(
                f"Could not find a template with {template_id} in registry."
            )
        return templates[0]

    def __str__(self):
        templates_infos = [
            {
                "name": t.name,
                "id": t.model_template_id,
                "path": t.model_template_path,
                "task_type": str(t.task_type),
            }
            for t in self.templates
        ]
        return yaml.dump(templates_infos)


def find_and_parse_model_template(path_or_id):
    """
    In first function attempts to read a model template from disk under assumption that a path is passed.
    If the attempt is failed, it tries to find template in registry under assumption that an ID is passed.
    """

    if os.path.exists(path_or_id):
        return parse_model_template(path_or_id)
    return Registry(".").get(path_or_id)
