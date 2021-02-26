import glob
import logging
import os
import sys
from subprocess import run
import tempfile
import unittest

import yaml

ENABLE_TESTS_FOR = {
    'object_detection': [
        'custom_object_detection',
        'face_detection',
        'person_detection',
        'person_vehicle_bike_detection',
        'text_detection',
        'vehicle_detection',
    ],
    'action_recognition_2': [
        'gesture-recognition',
    ],
    'instance_segmentation': [
        'coco_instance_segmentation',
    ],
    'text_spotting': [
        'alphanumeric_text_spotting',
    ],
    'image_classification': [
        'custom_classification',
    ],
}

NNCF_ENABLE_FOR = [
    'object_detection',
]

ENABLE_TRAIN_TESTS = True
ENABLE_EXPORT_TESTS = True
ENABLE_NNCF_TESTS = True

def _is_verbose_flag_set():
    if '-v' in sys.argv or '--verbose' in sys.argv:
        return True
    else:
        return False

def run_with_log(*args, **kwargs):
    cmd = args[0]
    logging.info(f'Running command\n`{cmd}`')
    return run(*args, **kwargs)

def create_model_template_tests_base(subfolder_name):
    class _ModelTemplatesTestBase(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            logging.basicConfig(level=logging.INFO)
            verbose = _is_verbose_flag_set()
            if verbose:
                logging.info('Running with verbosity=True')

            workdir = os.environ.get('WORKDIR')
            if not workdir:
                cls.work_dir = os.path.join(tempfile.mkdtemp(), subfolder_name)
            else:
                if not os.path.isabs(workdir):
                    raise RuntimeError(f'Received non-absolute path as WORKDIR="{workdir}" -- it is not reliable')
                cls.work_dir = os.path.join(os.path.abspath(workdir), subfolder_name)

            templates_filter_environ = os.environ.get('TEMPLATES_FILTER')
            if templates_filter_environ:
                templates_filter_arg = f'--templates-filter "{templates_filter_environ}"'
            else:
                templates_filter_arg = ''

            if verbose:
                cls.verbosity_flag = '--verbose'
            else:
                cls.verbosity_flag = ''

            run_with_log(f'python3 tools/instantiate.py {templates_filter_arg} --do-not-load-snapshots {cls.verbosity_flag} {cls.work_dir}', shell=True, check=True)

        def _get_template_files(self):
            template_filenames = glob.glob(f'{self.work_dir}/**/template.yaml', recursive=True)
            template_filenames = list(template_filenames)
            return template_filenames

    return _ModelTemplatesTestBase

class ModelTemplatesTestCase(create_model_template_tests_base('TESTS')):
    def test_existence_of_mandatory_files_in_template_dir(self):
        template_files = self._get_template_files()
        for template in template_files:
            template_dirname = os.path.dirname(template)
            self.assertTrue(os.path.exists(os.path.join(template_dirname, 'train.py')))
            self.assertTrue(os.path.exists(os.path.join(template_dirname, 'eval.py')))
            self.assertTrue(os.path.exists(os.path.join(template_dirname, 'export.py')))
            self.assertTrue(os.path.exists(os.path.join(template_dirname, 'compress.py')))
            self.assertTrue(os.path.exists(os.path.join(template_dirname, 'requirements.txt')))
            self.assertTrue(os.path.exists(os.path.join(template_dirname, 'modules.yaml')))

    def test_existence_of_mandatory_fields_in_template(self):
        mandatory_fields = [
            'name',
            'domain',
            'problem',
            'framework',
            'summary',
            'annotation_format',
            'initial_weights',
            'dependencies',
            'max_nodes',
            'training_target',
            'inference_target',
            'hyper_parameters',
            'output_format',
            'optimisations',
            'metrics',
            'tensorboard',
        ]
        template_files = self._get_template_files()
        for template in template_files:
            with open(template) as read_file:
                template_fields = list(yaml.safe_load(read_file).keys())
            max_pos = -1
            for field in mandatory_fields:
                try:
                    pos = template_fields.index(field)
                except ValueError as e:
                    logging.error(f'Failed to find "{field}" in {template}')
                    raise e
                assert pos > max_pos, f'Unsorted fields in template file: {template}'
                max_pos = pos

    def test_train_and_eval(self):
        if not ENABLE_TRAIN_TESTS:
            return

        template_files = self._get_template_files()
        domain_folders = set()
        for template_file in template_files:
            with open(template_file) as read_file:
                domain_folder = yaml.load(read_file, yaml.SafeLoader)['domain'].lower().replace(' ', '_')
                domain_folders.add(domain_folder)

        for domain_folder in domain_folders:
            if domain_folder not in ENABLE_TESTS_FOR:
                continue

            venv_activate_path = os.path.join(self.work_dir, domain_folder, 'venv', 'bin', 'activate')
            for problem_folder in ENABLE_TESTS_FOR[domain_folder]:
                logging.info(f'Running tests for {domain_folder}/{problem_folder}.')
                returncode = run_with_log(
                    f'. {venv_activate_path};'
                    f'export MODEL_TEMPLATES={self.work_dir};'
                    f'python3 {os.path.join(domain_folder, "tests", "run_train_tests.py")}'
                    f' --pattern=train_tests_{problem_folder}.py {self.verbosity_flag}',
                    shell=True,
                    check=True,
                    executable="/bin/bash").returncode
                self.assertEqual(returncode, 0)

    def test_export(self):
        if not ENABLE_EXPORT_TESTS:
            return

        template_files = self._get_template_files()
        domain_folders = set()
        for template_file in template_files:
            with open(template_file) as read_file:
                domain_folder = yaml.load(read_file, yaml.SafeLoader)['domain'].lower().replace(' ', '_')
                domain_folders.add(domain_folder)

        for domain_folder in domain_folders:
            if domain_folder not in ENABLE_TESTS_FOR:
                continue

            venv_activate_path = os.path.join(self.work_dir, domain_folder, 'venv', 'bin', 'activate')
            for problem_folder in ENABLE_TESTS_FOR[domain_folder]:
                logging.info(f'Running export tests for {domain_folder}/{problem_folder}.')
                returncode = run_with_log(
                    f'. {venv_activate_path};'
                    f'export MODEL_TEMPLATES={self.work_dir};'
                    f'python3 {os.path.join(domain_folder, "tests", "run_export_tests.py")}'
                    f' --pattern=export_tests_{problem_folder}.py {self.verbosity_flag}',
                    shell=True,
                    check=True,
                    executable="/bin/bash").returncode
                self.assertEqual(returncode, 0)

class ModelTemplatesNNCFTestCase(create_model_template_tests_base('NNCF_TESTS')):
    """
    Note that for NNCF tests we create a separate folder, since NNCF tests use
    each of the folders created by tools/instantiate.py as the base for many
    other tests.
    That is, each of the compression tests does as follows:
    * copies a folder with a model template to a folder with a new name,
    * then makes changes in template.yaml and other model template files
      in the copied folder
    * then runs training/testing/etc.
    Although the original folder of a model template is stayed unchanged by these
    operations, other tests (that are not related to NNCF) still may change it.
    So, it is better to have the original folders created by tools/instantiate.py
    clean to avoid side effects between tests.
    This is the reason why we use a separate test case with a separate
    subfolder for instantiating.
    """

    @classmethod
    def setUpClass(cls):
        if not ENABLE_NNCF_TESTS:
            return
        super().setUpClass()

    def test_nncf(self):
        # TODO(LeonidBeynenson) : move some big part of the methods to a common function
        if not ENABLE_NNCF_TESTS:
            return

        template_files = self._get_template_files()
        domain_folders = set()
        for template_file in template_files:
            with open(template_file) as read_file:
                domain_folder = yaml.load(read_file, yaml.SafeLoader)['domain'].lower().replace(' ', '_')
                domain_folders.add(domain_folder)

        for domain_folder in domain_folders:
            if domain_folder not in ENABLE_TESTS_FOR or domain_folder not in NNCF_ENABLE_FOR:
                continue

            venv_activate_path = os.path.join(self.work_dir, domain_folder, 'venv', 'bin', 'activate')
            for problem_folder in ENABLE_TESTS_FOR[domain_folder]:
                logging.info(f'Running nncf tests for {domain_folder}/{problem_folder}.')
                returncode = run_with_log(
                    f'. {venv_activate_path};'
                    f'export MODEL_TEMPLATES={self.work_dir};'
                    f'python3 {os.path.join(domain_folder, "tests", "run_nncf_tests.py")}'
                    f' --pattern=nncf_tests_{problem_folder}.py {self.verbosity_flag}',
                    shell=True,
                    check=True,
                    executable="/bin/bash").returncode
                self.assertEqual(returncode, 0)

if __name__ == '__main__':
    # TODO(LeonidBeynenson): instead of unittest.main()
    #     make a usual argparser with parameters like unittest.main()
    #     plus --workdir and --templates-filter
    #     instead of environment variables
    unittest.main()
