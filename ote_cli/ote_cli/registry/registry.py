import copy
import glob
import os

import yaml
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


    def filter(self, framework=None, domain=None):
        templates = copy.deepcopy(self.templates)
        if framework is not None:
            templates = [template for template in templates if template['framework'] == framework]
        if domain is not None:
            templates = [template for template in templates if template['task_type'] == domain]
        return Registry(templates=templates)

    def get(self, name):
        templates = [template for template in self.templates if template['name'] == name]
        return templates[0]


    def __repr__(self):
        return yaml.dump(self.templates)
