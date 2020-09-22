# Copyright (C) 2020 Intel Corporation
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

import json
import os
import unittest
import yaml
import tempfile

from common.utils import download_if_not_yet, collect_ap, run_through_shell


def get_dependencies(template_file):
    output = {}
    with open(template_file) as read_file:
        content = yaml.load(read_file, yaml.SafeLoader)
        for dependency in content['dependencies']:
            output[dependency['destination'].split('.')[0]] = dependency['source']
        return output


def get_epochs(template_file):
    with open(template_file) as read_file:
        content = yaml.load(read_file, yaml.SafeLoader)
    return content['hyper_parameters']['basic']['epochs']


def create_test_case(problem_name, model_name, ann_file, img_root):
    class TestCaseOteApi(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            cls.template_file = f'model_templates/{problem_name}/{model_name}/template.yaml'
            cls.ann_file = ann_file
            cls.img_root = img_root
            cls.work_dir = tempfile.mkdtemp()
            cls.dependencies = get_dependencies(cls.template_file)
            cls.epochs_delta = 3
            cls.total_epochs = get_epochs(cls.template_file) + cls.epochs_delta

            download_if_not_yet(cls.work_dir, cls.dependencies['snapshot'])

        def test_evaluation(self):
            log_file = os.path.join(self.work_dir, 'test_evaluation.log')
            run_through_shell(
                f'cd {os.path.dirname(self.template_file)};'
                f'python {self.dependencies["eval"]}'
                f' --test-ann-files {self.ann_file}'
                f' --test-img-roots {self.img_root}'
                f' --save-metrics-to {os.path.join(self.work_dir, "metrics.yaml")}'
                f' --load-weights {os.path.join(self.work_dir, os.path.basename(self.dependencies["snapshot"]))}'
                f' | tee {log_file}')

            with open(os.path.join(self.work_dir, "metrics.yaml")) as read_file:
                content = yaml.load(read_file, yaml.SafeLoader)

            ap = [metrics['value']
                  for metrics in content['metrics'] if metrics['key'] == 'ap'][0]

            with open(f'tests/expected_outputs/{problem_name}/{model_name}.json') as read_file:
                content = json.load(read_file)

            self.assertLess(abs(content['map'] - ap / 100), 1e-6)

        def test_finetuning(self):
            log_file = os.path.join(self.work_dir, 'test_finetuning.log')
            run_through_shell(
                f'cd {os.path.dirname(self.template_file)};'
                f'python {self.dependencies["train"]}'
                f' --train-ann-files {self.ann_file}'
                f' --train-img-roots {self.img_root}'
                f' --val-ann-files {self.ann_file}'
                f' --val-img-roots {self.img_root}'
                f' --resume-from {os.path.join(self.work_dir, os.path.basename(self.dependencies["snapshot"]))}'
                f' --save-checkpoints-to {self.work_dir}'
                f' --gpu-num 1'
                f' --batch-size 1'
                f' --epochs {self.total_epochs}'
                f' | tee {log_file}')

            ap = collect_ap(log_file)
            self.assertEqual(len((ap)), self.epochs_delta)
            self.assertLess(ap[0] * 0.9, ap[-1])

    return TestCaseOteApi


def create_export_test_case(problem_name, model_name, ann_file, img_root, alt_ssd_export=False):
    class ExportTestCase(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            cls.template_file = f'model_templates/{problem_name}/{model_name}/template.yaml'
            cls.work_dir = tempfile.mkdtemp()
            cls.dependencies = get_dependencies(cls.template_file)

            download_if_not_yet(cls.work_dir, cls.dependencies['snapshot'])

            cls.test_export_thr = 0.031

            run_through_shell(
                f'cd {os.path.dirname(cls.template_file)};'
                f'/opt/intel/openvino/bin/setupvars.sh;'
                f'python {cls.dependencies["export"]}'
                f' --load-weights {os.path.join(cls.work_dir, os.path.basename(cls.dependencies["snapshot"]))}'
                f' --save-model-to {os.path.join(cls.work_dir, "export")}'
            )

        def export_test(self, alt_ssd_export, thr):
            if alt_ssd_export:
                export_dir = os.path.join(self.work_dir, "export", "alt_ssd_export")
                log_file = os.path.join(export_dir, 'test_alt_ssd_export.log')
            else:
                export_dir = os.path.join(self.work_dir, "export")
                log_file = os.path.join(export_dir, 'test_export.log')

            run_through_shell(
                f'cd {os.path.dirname(self.template_file)};'
                f'/opt/intel/openvino/bin/setupvars.sh;'
                f'python {self.dependencies["eval"]}'
                f' --test-ann-files {ann_file}'
                f' --test-img-roots {img_root}'
                f' --load-weights {os.path.join(export_dir, "model.bin")}'
                f' --save-metrics-to {os.path.join(export_dir, "metrics.yaml")}'
            )

            with open(os.path.join(export_dir, "metrics.yaml")) as read_file:
                content = yaml.load(read_file, yaml.SafeLoader)
                ap = [metric for metric in content['metrics'] if metric['key'] == 'ap'][0]['value']

            with open(f'tests/expected_outputs/{problem_name}/{model_name}.json') as read_file:
                content = json.load(read_file)

            self.assertGreater(ap, content['map'] - thr)

        def test_export(self):
            self.export_test(False, self.test_export_thr)

    class ExportWithAltSsdTestCase(ExportTestCase):

        def test_alt_ssd_export(self):
            self.export_test(True, self.test_export_thr)

    if alt_ssd_export:
        return ExportWithAltSsdTestCase

    return ExportTestCase
