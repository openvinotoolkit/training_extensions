# Copyright (C) 2020-2021 Intel Corporation
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

import logging
import os
import re
import pathlib
import yaml

from pprint import pformat

from ote.tests.test_case import (create_export_test_case,
                                 create_test_case,
                                 create_nncf_test_case,
                                 skip_if_cuda_not_available)
from ote.tests.utils import find_files_by_pattern, extract_last_lines_by_pattern
from ote.utils.misc import run_through_shell

def create_image_classification_export_test_case(**kwargs):
    expected_outputs_dir = os.path.join(os.path.dirname(__file__), '..', 'expected_outputs')
    ExportTestCase = create_export_test_case('image_classification',
                                             **kwargs,
                                             metric_keys=['accuracy'],
                                             expected_outputs_dir=expected_outputs_dir)

    class ClassificationExportTestCase(ExportTestCase):
        def test_export_on_gpu(self):
            skip_if_cuda_not_available()
            export_dir = os.path.join(self.output_folder, 'gpu_export')
            self.do_export(export_dir, on_gpu=True)

        def test_export_on_cpu(self):
            export_dir = os.path.join(self.output_folder, 'cpu_export')
            self.do_export(export_dir, on_gpu=False)

        def do_export(self, export_dir, on_gpu):
            if not os.path.exists(export_dir):
                initial_command = 'export CUDA_VISIBLE_DEVICES=;' if not on_gpu else ''
                run_through_shell(
                    f'{initial_command}'
                    f'cd {os.path.dirname(self.template_file)};'
                    f'pip install -r requirements.txt;'
                    f'python3 export.py --openvino'
                    f' --load-weights snapshot.pth'
                    f' --save-model-to {export_dir}'
                )
                self.assertTrue(len(list(pathlib.Path(export_dir).rglob('*.onnx'))) > 0, 'Export to onnx failed')
                self.assertTrue(len(list(pathlib.Path(export_dir).rglob('*.bin'))) > 0, 'Export to openvino failed')

    return ClassificationExportTestCase


def create_image_classification_test_case(**kwargs):
    expected_outputs_dir = os.path.join(os.path.dirname(__file__), '..', 'expected_outputs')
    TrainTestCase = create_test_case('image_classification',
                                     **kwargs,
                                     metric_keys=['accuracy'],
                                     expected_outputs_dir=expected_outputs_dir,
                                     batch_size=2)
    class ClassificationTrainTestCase(TrainTestCase):
        def do_finetuning(self, on_gpu):
            self.total_epochs = 5
            log_file = os.path.join(self.output_folder, 'test_finetuning.log')
            initial_command = 'export CUDA_VISIBLE_DEVICES=;' if not on_gpu else ''
            run_through_shell(
                f'{initial_command}'
                f'cd {self.template_folder};'
                f'python3 train.py'
                f' --train-ann-files {self.ann_file}'
                f' --train-data-roots {os.path.join(self.img_root, "train")}'
                f' --val-ann-files {self.ann_file}'
                f' --val-data-roots {os.path.join(self.img_root, "val")}'
                f' --load-weights snapshot.pth'
                f' --save-checkpoints-to {self.output_folder}'
                f' --gpu-num 1'
                f' --batch-size {self.batch_size}'
                f' --epochs {self.total_epochs}'
                f' | tee {log_file}')

            self.assertTrue(os.path.exists(os.path.join(self.output_folder, 'latest.pth')))
    return ClassificationTrainTestCase

def make_field_value_change_in_struct_recursively(struct, field_name, new_field_value):
    """
    This function makes recursive walk in the structure `struct` and for each dict
    if the dict has key `field_name`, the function sets its value
    to `new_field_value` (may be any python structure)
    """
    logging.debug(f'Making recursive change in struct: {field_name}: {new_field_value}')
    def _make_change(cur_struct, field_name, new_field_value, set_ids=None):
        if set_ids is None:
            set_ids = set()
        if id(cur_struct) in set_ids:
            return
        set_ids.add(id(cur_struct))
        if isinstance(cur_struct, dict):
            for k in sorted(cur_struct.keys()):
                _make_change(cur_struct[k], field_name, new_field_value, set_ids)
            if field_name in cur_struct:
                cur_struct[field_name] = new_field_value
            return
        if isinstance(cur_struct, list):
            # pylint: disable=consider-using-enumerate
            for i in range(len(cur_struct)):
                _make_change(cur_struct[i], field_name, new_field_value, set_ids)
            return
        # if neither dict nor list -- nothing to do
        return

    _make_change(struct, field_name, new_field_value)

def make_field_value_changes_in_struct_recursively(struct, field_value_changes_in_struct):
    logging.debug(f'Before recursive changes in struct:\n{pformat(struct)}')
    for field_name, new_field_value in sorted(field_value_changes_in_struct.items()):
        make_field_value_change_in_struct_recursively(struct, field_name, new_field_value)
    logging.debug(f'After recursive changes in struct:\n{pformat(struct)}')

def create_image_classification_nncf_test_case(problem_name, model_name, ann_file, img_root,
                                               compression_cmd_line_parameters,
                                               template_update_dict=None,
                                               compression_cfg_update_dict=None,
                                               field_value_changes_in_nncf_config=None):
    """
    Note the field_value_changes_in_nncf_config should be a dict
    value that should be applied to make_field_value_changes_in_struct_recursively
    """
    # pylint: disable=too-many-arguments, too-many-statements
    if template_update_dict is None:
        template_update_dict = {}

    NNCFBaseTestCase = create_nncf_test_case('image_classification', problem_name, model_name, ann_file, img_root,
                                             compression_cmd_line_parameters,
                                             template_update_dict=template_update_dict,
                                             compression_cfg_update_dict=compression_cfg_update_dict)

    class ClassificationNNCFTestCase(NNCFBaseTestCase):
        @classmethod
        def setUpClass(cls):
            super(ClassificationNNCFTestCase, cls).setUpClass()
            if field_value_changes_in_nncf_config:
                cls._apply_field_value_changes_in_nncf_config_recursively(cls.template_file,
                                                                          field_value_changes_in_nncf_config)

        @staticmethod
        def _apply_field_value_changes_in_nncf_config_recursively(template_file, field_value_changes_in_nncf_config):
            import mmcv
            from ote.modules.compression import get_optimisation_config_from_template

            assert isinstance(field_value_changes_in_nncf_config, dict)

            template_data = mmcv.Config.fromfile(template_file)
            compression_cfg_rel_path = get_optimisation_config_from_template(template_data)
            compression_cfg_path = os.path.join(os.path.dirname(template_file), compression_cfg_rel_path)

            compression_cfg = mmcv.Config.fromfile(compression_cfg_path)
            compression_cfg = dict(compression_cfg)

            make_field_value_changes_in_struct_recursively(compression_cfg, field_value_changes_in_nncf_config)

            mmcv.dump(compression_cfg, compression_cfg_path, file_format='json')
            logging.debug(f'Made new compression config file:\n{pformat(mmcv.load(compression_cfg_path))}')

        def setUp(self):
            super().setUp()
            self.preliminary_training_folder = os.path.join(self.template_folder, 'preliminary_training')
            os.makedirs(self.preliminary_training_folder, exist_ok=True)
            self.batch_size = 2

        @staticmethod
        def _find_best_models(folder_path):
            best_models = find_files_by_pattern(folder_path, '*best*.pth*.tar*')
            return best_models

        def _find_latest_model(self, folder_path):
            latest_models = find_files_by_pattern(folder_path, 'latest.*pth*')
            self.assertEqual(len(latest_models), 1)
            return latest_models[0]

        def do_preliminary_finetuning(self, on_gpu):
            logging.info(f'Looking for best models in {self.preliminary_training_folder}')
            best_models = self._find_best_models(self.preliminary_training_folder)
            if best_models:
                logging.info(f'Found already generated by preliminary finetuning the best models {best_models}')
                return best_models

            logging.info(f'Begin making preliminary finetuning for the best models {best_models}')
            self.total_epochs = 5
            log_file = os.path.join(self.output_folder, 'test_finetuning.log')
            initial_command = 'export CUDA_VISIBLE_DEVICES=;' if not on_gpu else ''
            run_through_shell(
                f'{initial_command}'
                f'cd {self.template_folder};'
                f'python3 train.py'
                f' --train-ann-files {self.ann_file}'
                f' --train-data-roots {os.path.join(self.img_root, "train")}'
                f' --val-ann-files {self.ann_file}'
                f' --val-data-roots {os.path.join(self.img_root, "val")}'
                f' --load-weights snapshot.pth'
                f' --save-checkpoints-to {self.preliminary_training_folder}'
                f' --gpu-num 1'
                f' --batch-size {self.batch_size}'
                f' --epochs {self.total_epochs}'
                f' | tee {log_file}')
            logging.info(f'End making preliminary finetuning for the best models {best_models}')

            self.assertTrue(os.path.exists(os.path.join(self.preliminary_training_folder, 'latest.pth')))
            best_models = self._find_best_models(self.preliminary_training_folder)
            logging.info(f'Found best models in {self.preliminary_training_folder}\n best_models={best_models}')
            self.assertGreater(len(best_models), 0)
            logging.info(f'Generated best_models = {best_models}')
            return best_models

        def do_compress(self, main_weights_path, aux_weights_path):
            log_file = os.path.join(self.output_folder, f'log__{self.id()}.txt')
            run_through_shell(
                f'cd {self.template_folder};'
                f'python3 compress.py'
                f' --train-ann-files {self.ann_file}'
                f' --train-data-roots {os.path.join(self.img_root, "train")}'
                f' --val-ann-files {self.ann_file}'
                f' --val-data-roots {os.path.join(self.img_root, "val")}'
                f' --save-checkpoints-to {self.output_folder}'
                f' --gpu-num 1'
                f' --load-weights {main_weights_path} --load-aux-weights {aux_weights_path}'
                + ' ' + self.compress_cmd_line_params
                + f' | tee {log_file}')
            return log_file

        def do_eval(self, file_to_eval):
            metrics_path = os.path.join(self.output_folder, 'metrics.yaml')
            run_through_shell(
                f'cd {self.template_folder};'
                f'python3 eval.py'
                f' --test-ann-files {self.ann_file}'
                f' --test-data-roots {os.path.join(self.img_root, "val")}'
                f' --save-metrics-to {metrics_path}'
                f' --load-weights {file_to_eval}'
                )
            return metrics_path

        def test_nncf_compress_on_gpu(self):
            skip_if_cuda_not_available()
            logging.info('Begin test_nncf_compress_on_gpu')
            best_models = self.do_preliminary_finetuning(True)
            self.assertEqual(len(best_models), 2)
            self.assertIn('main_model', best_models[1])
            self.assertIn('aux_model_1', best_models[0])

            log_file = self.do_compress(main_weights_path=best_models[0],
                                        aux_weights_path=best_models[1])
            logging.debug('Compression is finished')
            best_compressed_models = self._find_best_models(self.output_folder)
            logging.debug(f'Found best compressed models: {best_compressed_models}')
            self.assertEqual(len(best_compressed_models), 2)
            self.assertIn('main_model', best_compressed_models[1])
            self.assertIn('aux_model_1', best_compressed_models[0])

            logging.info('End test_nncf_compress_on_gpu')
            return log_file

        def _extract_last_rank1_from_log(self, log_path):
            re_rank1 = re.compile('^.*Rank-1 *: *')
            two_last_rank1 = extract_last_lines_by_pattern(log_path, re_rank1, 2)
            self.assertEqual(len(two_last_rank1), 2)
            main_rank1_line = two_last_rank1[0]
            last_rank1 = re.sub(re_rank1, '', main_rank1_line).strip()
            self.assertEqual(last_rank1[-1], '%', msg=f'Wrong Rank-1 line {main_rank1_line}')
            last_rank1 = last_rank1.strip('%')
            logging.debug(f'Found last rank1 in log: rank1 = "{last_rank1}" %')
            return float(last_rank1) / 100.0

        def _extract_accuracy_from_metrics_file(self, metrics_path):
            with open(metrics_path) as f:
                metrics = yaml.safe_load(f)
            metrics = metrics['metrics']
            accuracy_metric = [m for m in metrics if isinstance(m, dict) and m.get('key') == 'accuracy']
            self.assertEqual(len(accuracy_metric), 1, msg=f'Wrong file {metrics_path}')
            accuracy_metric = accuracy_metric[0]
            accuracy = accuracy_metric['value']
            if accuracy_metric.get('unit') == '%':
                accuracy = accuracy / 100.0

            return accuracy

        def test_nncf_compress_and_eval_on_gpu(self):
            skip_if_cuda_not_available()
            logging.info('Begin test_nncf_compress_and_eval_on_gpu')
            best_models = self.do_preliminary_finetuning(True)
            self.assertEqual(len(best_models), 2)
            self.assertIn('main_model', best_models[1])
            self.assertIn('aux_model_1', best_models[0])

            log_file = self.do_compress(main_weights_path=best_models[0],
                                        aux_weights_path=best_models[1])
            logging.debug('Compression is finished')
            latest_compressed_model = self._find_latest_model(self.output_folder)
            logging.debug(f'Found latest compressed models: {latest_compressed_model}')

            last_training_rank1 = self._extract_last_rank1_from_log(log_file)

            logging.debug(f'Before making evaluation of {latest_compressed_model}')
            metrics_path = self.do_eval(latest_compressed_model)
            logging.debug(f'After making evaluation of {latest_compressed_model}')
            logging.debug(f'Metrics are stored in {metrics_path}')

            accuracy = self._extract_accuracy_from_metrics_file(metrics_path)
            self.assertAlmostEqual(last_training_rank1, accuracy, delta=0.001,
                                   msg=f'Difference between accuracy from log file {log_file} '
                                       f'and the accuracy from evaluation metrics file {metrics_path}')
            logging.info('End test_nncf_compress_and_eval_on_gpu')

            return log_file, metrics_path

        def test_nncf_compress_and_export(self):
            skip_if_cuda_not_available()
            logging.info('Begin test_nncf_compress_and_export')
            best_models = self.do_preliminary_finetuning(True)
            self.assertEqual(len(best_models), 2)
            self.assertIn('main_model', best_models[1])
            self.assertIn('aux_model_1', best_models[0])

            log_file = self.do_compress(main_weights_path=best_models[0],
                                        aux_weights_path=best_models[1])
            logging.debug('Compression is finished')
            latest_compressed_model = self._find_latest_model(self.output_folder)
            logging.debug(f'Found latest compressed models: {latest_compressed_model}')

            logging.info('Exporting the latest compressed model')
            export_dir = self.output_folder
            run_through_shell(
                f'cd {os.path.dirname(self.template_file)};'
                f'python3 export.py --openvino'
                f' --load-weights {latest_compressed_model}'
                f' --save-model-to {export_dir}'
            )
            onnx_res_files = find_files_by_pattern(export_dir, '*.onnx')
            xml_res_files = find_files_by_pattern(export_dir, '*.xml')
            bin_res_files = find_files_by_pattern(export_dir, '*.bin')
            self.assertTrue(len(onnx_res_files) == 1, 'Export to onnx failed')
            self.assertTrue(len(xml_res_files) == 1, 'Export to openvino failed')
            self.assertTrue(len(bin_res_files) == 1, 'Export to openvino failed')

            xml_res_file = xml_res_files[0]
            logging.debug(f'Before making evaluation of {xml_res_file}')
            metrics_path = self.do_eval(xml_res_file)
            logging.debug(f'After making evaluation of {xml_res_file}')
            logging.debug(f'    metrics are stored to the file {metrics_path}')

            logging.info('End test_nncf_compress_and_export')

            return log_file, metrics_path

    return ClassificationNNCFTestCase
