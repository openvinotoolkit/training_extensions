import copy
import glob
import logging
import os

import yaml

from ote_sdk.entities.model_template import parse_model_template 

from ote_cli.utils.loading import load_config


class Registry:

    def __init__(self, templates_dir=None, templates=None):
        if templates is None:
            if templates_dir is None:
                templates_dir = os.getenv('TEMPLATES_DIR')

            if templates_dir is None:
                raise RuntimeError('The templates_dir is not set.')

            template_filenames = glob.glob(os.path.join(templates_dir, '**', 'template.yaml'), recursive=True)
            template_filenames = [os.path.abspath(p) for p in template_filenames]

            self.templates = []

            for template_file in template_filenames:
                config = load_config(template_file)
                self.templates.append({
                    'framework': config['framework'].replace(' ', '_'),
                    'task_type': config['task_type'].replace(' ', '_'),
                    'name': config['name'].replace(' ', '_'),
                    'path': template_file
                })
        else:
            self.templates = copy.deepcopy(templates)

        self.task_types = self.__collect_task_types(self.templates)

    @staticmethod
    def __collect_task_types(templates):
        return {template['task_type'] for template in templates}

    def filter(self, framework=None, task_type=None):
        templates = copy.deepcopy(self.templates)
        if framework is not None:
            templates = [template for template in templates if template['framework'].lower() == framework.lower()]
        if task_type is not None:
            templates = [template for template in templates if template['task_type'].lower() == task_type.lower()]
        return Registry(templates=templates)

    def get(self, name):
        templates = [template for template in self.templates if template['name'].lower() == name.lower()]
        return templates[0]

    def __repr__(self):
        return yaml.dump(self.templates)


def find_and_parse_model_template(path_or_name):
    if os.path.exists(path_or_name):
        return parse_model_template(path_or_name)
    template = [template for template in Registry('.').templates if template['name'] == path_or_name]
    if template:
        path = template[0]['path']
        logging.warning(f'Parsing model template {path_or_name}: {path}')
        return parse_model_template(path)
    raise ValueError(f'Could not find {path_or_name} in registry.')
