"""Model templates registry."""

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
from pathlib import Path
from typing import Optional

import yaml

from otx.api.entities.model_template import parse_model_template
from otx.cli.utils.importing import get_backbone_list, get_otx_root_path


class Registry:
    """Class that implements a model templates registry."""

    def __init__(self, templates_dir=None, templates=None, experimental=False):
        if templates is None:
            if templates_dir is None:
                templates_dir = os.getenv("TEMPLATES_DIR")

            if templates_dir is None:
                raise RuntimeError("The templates_dir is not set.")

            template_filenames = glob.glob(os.path.join(templates_dir, "**", "template.yaml"), recursive=True)
            if experimental:
                template_filenames.extend(
                    glob.glob(
                        os.path.join(templates_dir, "**", "template_experimental.yaml"),
                        recursive=True,
                    )
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
        """Filters registry by framework and/or task type and returns filtered registry."""

        templates = copy.deepcopy(self.templates)
        if framework is not None:
            templates = [template for template in templates if template.framework.lower() == framework.lower()]
        if task_type is not None:
            templates = [template for template in templates if str(template.task_type).lower() == task_type.lower()]
        return Registry(templates=templates)

    def get(self, template_id, skip_error=False):
        """Returns a model template with specified template_id or template.name."""

        templates = [
            template
            for template in self.templates
            if str(template_id).upper() in (str(template.model_template_id).upper(), str(template.name).upper())
        ]
        if not templates:
            if skip_error:
                return None
            raise ValueError(f"Could not find a template with {template_id} in registry.")
        return templates[0]

    def get_backbones(self, backend_list):
        """Returns list of backbones for a given template."""
        backbone_list = {}
        for backend in backend_list:
            backbone_list[backend] = get_backbone_list(backend)
        return backbone_list

    def __str__(self):
        """Returns the string representation of the registry."""
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
    """In first function attempts to read a model template from disk under assumption that a path is passed.

    If the attempt is failed, it tries to find template in registry under assumption that an ID or name is passed.
    """
    # Return None Type
    if not path_or_id:
        return path_or_id

    # 1. Find from path
    if is_template(path_or_id):
        return parse_model_template(path_or_id)
    # 2. Find from id or Name
    return Registry(get_otx_root_path()).get(path_or_id, skip_error=True)


def is_template(template_path: Optional[str]) -> bool:
    """A function that determines whether the corresponding template path is a template.

    Args:
        template_path (str): The path of the file you want to know if it is a template.

    Returns:
        bool: True if template_path is template file else False.
    """
    if template_path and Path(template_path).is_file() and "template" in Path(template_path).name:
        return True
    return False
