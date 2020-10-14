import glob
import os
from subprocess import run
import sys
import tempfile
import unittest

import yaml


class ModelTemplatesTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        venv_dir = tempfile.mkdtemp()
        cls.work_dir = tempfile.mkdtemp()
        commands = [
            f'virtualenv -p python3 {venv_dir}',
            f'. {venv_dir}/bin/activate',
            'pip install -r ote/requirements.txt',
            'pip install -e ote/',
            f'python tools/instantiate.py {cls.work_dir}'
        ]
        run(';'.join(commands), shell=True, check=True)

    def test_existance_of_mandatory_files_in_template_dir(self):
        template_files = glob.glob(f'{self.work_dir}/**/template.yaml', recursive=True)
        for template in template_files:
            template_dirname = os.path.dirname(template)
            self.assertTrue(os.path.exists(os.path.join(template_dirname, 'train.py')))
            self.assertTrue(os.path.exists(os.path.join(template_dirname, 'eval.py')))
            self.assertTrue(os.path.exists(os.path.join(template_dirname, 'export.py')))
            self.assertTrue(os.path.exists(os.path.join(template_dirname, 'quantize.py')))
            self.assertTrue(os.path.exists(os.path.join(template_dirname, 'requirements.txt')))
            self.assertTrue(os.path.exists(os.path.join(template_dirname, 'snapshot.pth')))

    def test_train_and_eval(self):
        template_files = glob.glob(f'{self.work_dir}/**/template.yaml', recursive=True)
        domain_folders = set()
        for template_file in template_files:
            with open(template_file) as read_file:
                domain_folder = yaml.load(read_file, yaml.SafeLoader)['domain'].lower().replace(' ', '_')
                domain_folders.add(domain_folder)

        for domain_folder in domain_folders:
            run(f'export MODEL_TEMPLATES={self.work_dir};'
                f'python3 {os.path.join(domain_folder, "tests", "run_train_tests.py")}', shell=True)

    def test_export(self):
        template_files = glob.glob(f'{self.work_dir}/**/template.yaml', recursive=True)
        domain_folders = set()
        for template_file in template_files:
            with open(template_file) as read_file:
                domain_folder = yaml.load(read_file, yaml.SafeLoader)['domain'].lower().replace(' ', '_')
                domain_folders.add(domain_folder)

        for domain_folder in domain_folders:
            run(f'export MODEL_TEMPLATES={self.work_dir};'
                f'python3 {os.path.join(domain_folder, "tests", "run_export_tests.py")}', shell=True)


def main():
    suite = unittest.TestSuite((ModelTemplatesTestCase(), ))
    ret = not unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful()
    sys.exit(ret)


if __name__ == '__main__':
    main()
