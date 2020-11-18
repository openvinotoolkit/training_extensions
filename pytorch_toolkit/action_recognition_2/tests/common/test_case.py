"""
 Copyright (c) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import json
import os
import unittest
from subprocess import run

import torch
import yaml

from ote.utils.misc import download_snapshot_if_not_yet


def run_through_shell(cmd):
    run(cmd, shell=True, check=True, executable="/bin/bash")


def collect_accuracy(path):
    accuracies = []
    content = '/mean_top1_acc:'
    with open(path) as input_stream:
        for line in input_stream:
            candidate = line.strip()
            if content in candidate:
                accuracies.append(float(candidate.split(' ')[-1]))

    return accuracies


def get_dependencies(template_file):
    output = {}
    with open(template_file) as read_file:
        content = yaml.load(read_file, yaml.SafeLoader)
        for dependency in content['dependencies']:
            output[dependency['destination'].split('.')[0]] = dependency['source']
        return output


def create_test_case(problem_name, model_name, ann_file, img_root):
    class TestCaseOteApi(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            cls.templates_folder = os.environ['MODEL_TEMPLATES']
            cls.template_folder = os.path.join(cls.templates_folder, 'action_recognition_2', problem_name, model_name)
            cls.template_file = os.path.join(cls.template_folder, 'template.yaml')
            cls.ann_file = ann_file
            cls.img_root = img_root
            cls.dependencies = get_dependencies(cls.template_file)

            download_snapshot_if_not_yet(cls.template_file, cls.template_folder)

            run_through_shell(
                f'cd {cls.template_folder};'
                f'pip install -r requirements.txt;'
            )

        def skip_if_cpu_is_not_supported(self):
            with open(self.template_file) as read_file:
                training_targets = [x.lower() for x in yaml.load(read_file, yaml.SafeLoader)['training_target']]
            if 'cpu' not in training_targets:
                self.skipTest('CPU is not supported.')

        @unittest.skipUnless(torch.cuda.is_available(), 'No GPU found')
        def test_evaluation_on_gpu(self):
            run_through_shell(
                f'cd {self.template_folder};'
                f'python eval.py'
                f' --test-ann-files {self.ann_file}'
                f' --test-data-roots {self.img_root}'
                f' --save-metrics-to metrics.yaml'
                f' --load-weights snapshot.pth'
            )

            with open(os.path.join(self.template_folder, "metrics.yaml")) as read_file:
                content = yaml.load(read_file, yaml.SafeLoader)

            est_accuracy = [metrics['value'] for metrics in content['metrics'] if metrics['key'] == 'accuracy'][0]

            with open(f'{os.path.dirname(__file__)}/../expected_outputs/{problem_name}/{model_name}.json') as read_file:
                content = json.load(read_file)
                ref_accuracy = content['accuracy']

            self.assertLess(abs(ref_accuracy - 1e-2 * est_accuracy), 1e-6)

        def test_evaluation_on_cpu(self):
            self.skip_if_cpu_is_not_supported()

            run_through_shell(
                'export CUDA_VISIBLE_DEVICES=;'
                f'cd {self.template_folder};'
                f'python eval.py'
                f' --test-ann-files {self.ann_file}'
                f' --test-data-roots {self.img_root}'
                f' --save-metrics-to metrics.yaml'
                f' --load-weights snapshot.pth'
            )

            with open(os.path.join(self.template_folder, "metrics.yaml")) as read_file:
                content = yaml.load(read_file, yaml.SafeLoader)

            est_accuracy = [metrics['value'] for metrics in content['metrics'] if metrics['key'] == 'accuracy'][0]

            with open(f'{os.path.dirname(__file__)}/../expected_outputs/{problem_name}/{model_name}.json') as read_file:
                content = json.load(read_file)
                ref_accuracy = content['accuracy']

            self.assertLess(abs(ref_accuracy - 1e-2 * est_accuracy), 1e-6)

        @unittest.skipUnless(torch.cuda.is_available(), 'No GPU found')
        def test_finetuning_on_gpu(self):
            log_file = os.path.join(self.template_folder, 'test_finetuning.log')
            run_through_shell(
                f'cd {self.template_folder};'
                f'python train.py'
                f' --train-ann-files {self.ann_file}'
                f' --train-data-roots {self.img_root}'
                f' --val-ann-files {self.ann_file}'
                f' --val-data-roots {self.img_root}'
                f' --load-weights snapshot.pth'
                f' --save-checkpoints-to {self.template_folder}'
                f' --gpu-num 1'
                f' --batch-size 2'
                f' --epochs 6'
                f' | tee {log_file}')

            accuracy = collect_accuracy(log_file)
            self.assertGreater(accuracy[-1], 0.0)

        def test_finetuning_on_cpu(self):
            self.skip_if_cpu_is_not_supported()

            log_file = os.path.join(self.template_folder, 'test_finetuning.log')
            run_through_shell(
                'export CUDA_VISIBLE_DEVICES=;'
                f'cd {self.template_folder};'
                f'python train.py'
                f' --train-ann-files {self.ann_file}'
                f' --train-data-roots {self.img_root}'
                f' --val-ann-files {self.ann_file}'
                f' --val-data-roots {self.img_root}'
                f' --load-weights snapshot.pth'
                f' --save-checkpoints-to {self.template_folder}'
                f' --gpu-num 1'
                f' --batch-size 2'
                f' --epochs 6'
                f' | tee {log_file}')

            accuracy = collect_accuracy(log_file)
            self.assertGreater(accuracy[-1], 0.0)

    return TestCaseOteApi


def create_export_test_case(problem_name, model_name, ann_file, img_root):
    class ExportTestCase(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            cls.templates_folder = os.environ['MODEL_TEMPLATES']
            cls.template_folder = os.path.join(cls.templates_folder, 'action_recognition_2', problem_name, model_name)
            cls.template_file = os.path.join(cls.template_folder, 'template.yaml')
            cls.ann_file = ann_file
            cls.img_root = img_root
            cls.dependencies = get_dependencies(cls.template_file)
            cls.test_export_thr = 1e-2

            download_snapshot_if_not_yet(cls.template_file, cls.template_folder)

            run_through_shell(
                f'cd {cls.template_folder};'
                f'pip install -r requirements.txt;'
            )

        def skip_if_cpu_is_not_supported(self):
            with open(self.template_file) as read_file:
                training_targets = [x.lower() for x in yaml.load(read_file, yaml.SafeLoader)['training_target']]
            if 'cpu' not in training_targets:
                self.skipTest('CPU is not supported.')

        def do_export(self, folder):
            run_through_shell(
                f'cd {os.path.dirname(self.template_file)};'
                f'pip install -r requirements.txt;'
                f'python export.py'
                f' --load-weights snapshot.pth'
                f' --save-model-to {folder}'
            )

        def export_test_on_gpu(self, thr):
            export_folder = 'gpu_export'
            if not os.path.exists(export_folder):
                self.do_export(export_folder)

            export_dir = os.path.join(self.template_folder, export_folder)

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
                est_accuracy = [metric['value'] for metric in content['metrics'] if metric['key'] == 'accuracy'][0]

            with open(f'{os.path.dirname(__file__)}/../expected_outputs/{problem_name}/{model_name}.json') as read_file:
                content = json.load(read_file)
                ref_accuracy = content['accuracy']

            self.assertGreater(1e-2 * est_accuracy, ref_accuracy - thr)

        def export_test_on_cpu(self, thr):
            export_folder = 'cpu_export'
            if not os.path.exists(export_folder):
                self.do_export(export_folder)

            export_dir = os.path.join(self.template_folder, export_folder)

            run_through_shell(
                f'export CUDA_VISIBLE_DEVICES=;'
                f'cd {os.path.dirname(self.template_file)};'
                f'python eval.py'
                f' --test-ann-files {ann_file}'
                f' --test-data-roots {img_root}'
                f' --load-weights {os.path.join(export_dir, "model.bin")}'
                f' --save-metrics-to {os.path.join(export_dir, "metrics.yaml")}'
            )

            with open(os.path.join(export_dir, "metrics.yaml")) as read_file:
                content = yaml.load(read_file, yaml.SafeLoader)
                est_accuracy = [metric['value'] for metric in content['metrics'] if metric['key'] == 'accuracy'][0]

            with open(f'{os.path.dirname(__file__)}/../expected_outputs/{problem_name}/{model_name}.json') as read_file:
                content = json.load(read_file)
                ref_accuracy = content['accuracy']

            self.assertGreater(1e-2 * est_accuracy, ref_accuracy - thr)

        @unittest.skipUnless(torch.cuda.is_available(), 'No GPU found')
        def test_export_on_gpu(self):
            self.export_test_on_gpu(self.test_export_thr)

        def test_export_on_cpu(self):
            self.skip_if_cpu_is_not_supported()
            self.export_test_on_cpu(self.test_export_thr)

    return ExportTestCase
