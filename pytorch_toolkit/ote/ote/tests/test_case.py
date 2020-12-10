import json
import os
import unittest

import torch
import yaml

from ote.tests.utils import collect_ap, run_through_shell
from ote.utils.misc import download_snapshot_if_not_yet


def get_dependencies(template_file):
    output = {}
    with open(template_file) as read_file:
        content = yaml.load(read_file, yaml.SafeLoader)
        for dependency in content['dependencies']:
            output[dependency['destination'].split('.')[0]] = dependency['source']
        return output


def get_epochs(template_file):
    with open(template_file) as read_file:
        content = yaml.safe_load(read_file)
    return content['hyper_parameters']['basic']['epochs']


def skip_non_instantiated_template_if_its_allowed(template_folder, problem_name, model_name):
    # Note that this is for debug purposes only
    # pylint: disable=no-else-raise
    should_skip_absent_templates = os.environ.get('SHOULD_SKIP_ABSENT_TEMPLATES')
    if not os.path.isdir(template_folder):
        if should_skip_absent_templates:
            raise unittest.SkipTest(f'The template folder for {problem_name}/{model_name}'
                                    f' is not instantiated -- SKIPPING IT')
        else:
            raise unittest.TestCase.failureException(f'The template folder for {problem_name}/{model_name}'
                                                     f' is not instantiated')


def skip_if_cpu_is_not_supported(template_file):
    with open(template_file) as read_file:
        training_targets = [x.lower() for x in yaml.safe_load(read_file)['training_target']]
    if 'cpu' not in training_targets:
        raise unittest.SkipTest('CPU is not supported.')


def create_test_case(domain_name, problem_name, model_name, ann_file, img_root, metric_keys, expected_outputs_dir):
    class TestCaseOteApi(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            cls.templates_folder = os.environ['MODEL_TEMPLATES']
            cls.template_folder = os.path.join(cls.templates_folder, domain_name, problem_name, model_name)
            skip_non_instantiated_template_if_its_allowed(cls.template_folder, problem_name, model_name)
            cls.template_file = os.path.join(cls.template_folder, 'template.yaml')
            cls.ann_file = ann_file
            cls.img_root = img_root
            cls.dependencies = get_dependencies(cls.template_file)
            cls.epochs_delta = 1
            cls.total_epochs = get_epochs(cls.template_file) + cls.epochs_delta

            expected_outputs_json = f'{expected_outputs_dir}/{problem_name}/{model_name}.json'

            with open(expected_outputs_json) as read_file:
                cls.expected_outputs = json.load(read_file)

            download_snapshot_if_not_yet(cls.template_file, cls.template_folder)

            run_through_shell(
                f'cd {cls.template_folder};'
                f'pip install -r requirements.txt;'
            )

        def do_evaluation(self, on_gpu):
            initial_command = 'export CUDA_VISIBLE_DEVICES=;' if not on_gpu else ''
            run_through_shell(
                f'{initial_command}'
                f'cd {self.template_folder};'
                f'python eval.py'
                f' --test-ann-files {self.ann_file}'
                f' --test-data-roots {self.img_root}'
                f' --save-metrics-to metrics.yaml'
                f' --load-weights snapshot.pth'
            )

            with open(os.path.join(self.template_folder, "metrics.yaml")) as read_file:
                content = yaml.load(read_file, yaml.SafeLoader)

            for metric_key in metric_keys:
                value = [metrics['value'] for metrics in content['metrics'] if metrics['key'] == metric_key][0]
                self.assertLess(abs(self.expected_outputs[metric_key] - value / 100), 1e-6)

        def do_finetuning(self, on_gpu):
            log_file = os.path.join(self.template_folder, 'test_finetuning.log')
            initial_command = 'export CUDA_VISIBLE_DEVICES=;' if not on_gpu else ''
            run_through_shell(
                f'{initial_command}'
                f'cd {self.template_folder};'
                f'python train.py'
                f' --train-ann-files {self.ann_file}'
                f' --train-data-roots {self.img_root}'
                f' --val-ann-files {self.ann_file}'
                f' --val-data-roots {self.img_root}'
                f' --resume-from snapshot.pth'
                f' --save-checkpoints-to {self.template_folder}'
                f' --gpu-num 1'
                f' --batch-size 1'
                f' --epochs {self.total_epochs}'
                f' | tee {log_file}')

            self.assertTrue(os.path.exists(os.path.join(self.template_folder, 'latest.pth')))

        @unittest.skipUnless(torch.cuda.is_available(), 'No GPU found')
        def test_evaluation_on_gpu(self):
            self.do_evaluation(on_gpu=True)

        def test_evaluation_on_cpu(self):
            skip_if_cpu_is_not_supported(self.template_file)
            self.do_evaluation(on_gpu=False)

        @unittest.skipUnless(torch.cuda.is_available(), 'No GPU found')
        def test_finetuning_on_gpu(self):
            self.do_finetuning(on_gpu=True)

        def test_finetuning_on_cpu(self):
            skip_if_cpu_is_not_supported(self.template_file)
            self.do_finetuning(on_gpu=False)
            

    return TestCaseOteApi


def create_export_test_case(domain_name, problem_name, model_name, ann_file, img_root, metric_keys, expected_outputs_dir):
    class ExportTestCase(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            cls.templates_folder = os.environ['MODEL_TEMPLATES']
            cls.template_folder = os.path.join(cls.templates_folder, domain_name, problem_name, model_name)
            skip_non_instantiated_template_if_its_allowed(cls.template_folder, problem_name, model_name)
            cls.template_file = os.path.join(cls.template_folder, 'template.yaml')
            cls.ann_file = ann_file
            cls.img_root = img_root
            cls.dependencies = get_dependencies(cls.template_file)
            cls.test_export_thr = 0.031

            expected_outputs_json = f'{expected_outputs_dir}/{problem_name}/{model_name}.json'

            with open(expected_outputs_json) as read_file:
                cls.expected_outputs = json.load(read_file)

            download_snapshot_if_not_yet(cls.template_file, cls.template_folder)


        def do_evaluation(self, export_dir):
            run_through_shell(
                f'cd {os.path.dirname(self.template_file)};'
                f'python eval.py'
                f' --test-ann-files {ann_file}'
                f' --test-data-roots {img_root}'
                f' --load-weights {os.path.join(export_dir, "model.bin")}'
                f' --save-metrics-to {os.path.join(export_dir, "metrics.yaml")}'
            )

            with open(os.path.join(export_dir, "metrics.yaml")) as read_file:
                content = yaml.load(read_file, yaml.SafeLoader)

            for metric_key in metric_keys:
                value = [metrics['value'] for metrics in content['metrics'] if metrics['key'] == metric_key][0]
                self.assertGreater(value, self.expected_outputs[metric_key] - self.test_export_thr)

        def do_export(self, export_dir, on_gpu):
            if not os.path.exists(export_dir):
                initial_command = 'export CUDA_VISIBLE_DEVICES=;' if not on_gpu else ''
                run_through_shell(
                    f'{initial_command}'
                    f'cd {os.path.dirname(self.template_file)};'
                    f'pip install -r requirements.txt;'
                    f'python export.py'
                    f' --load-weights snapshot.pth'
                    f' --save-model-to {export_dir}'
                )

        @unittest.skipUnless(torch.cuda.is_available(), 'No GPU found')
        def test_export_on_gpu(self):
            export_dir = os.path.join(self.template_folder, 'gpu_export')
            self.do_export(export_dir, on_gpu=True)
            self.do_evaluation(export_dir)

        def test_export_on_cpu(self):
            export_dir = os.path.join(self.template_folder, 'cpu_export')
            self.do_export(export_dir, on_gpu=False)
            self.do_evaluation(export_dir)

    return ExportTestCase
