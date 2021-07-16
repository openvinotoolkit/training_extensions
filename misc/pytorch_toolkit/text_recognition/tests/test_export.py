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

import os
import unittest
from copy import deepcopy
from tempfile import mkdtemp
import os.path

from text_recognition.utils.exporter import Exporter
from text_recognition.utils.evaluator import Evaluator, RunnerType
from text_recognition.utils.get_config import get_config
from text_recognition.utils.common import download_checkpoint

unittest.TestLoader.sortTestMethodsUsing = None


def create_export_test_case_for_monolithic(config_file, expected_outputs):
    class TestMonolithicExport(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            export_config = get_config(config_file, section='export')
            export_config['dataset'] = get_config(config_file, section='eval')['dataset']
            cls.config = export_config
            cls.config.update({'expected_outputs': expected_outputs})
            cls.model_path = os.path.join(mkdtemp(), os.path.split(cls.config.get('model_path'))[1])
            cls.res_model_name = os.path.join(os.path.dirname(cls.model_path), cls.config.get('res_model_name'))
            cls.config['res_model_name'] = cls.res_model_name
            cls.config['model_path'] = cls.model_path
            if not os.path.exists(cls.model_path):
                download_checkpoint(cls.model_path, cls.config.get('model_url'))
            cls.exporter = Exporter(cls.config)

        def test_complete_model_export(self):
            self.exporter.export_complete_model()
            result_model_exists = os.path.exists(self.res_model_name)
            self.assertEqual(True, result_model_exists)

        def test_onnx(self):
            self.exporter.export_complete_model()
            evaluator = Evaluator(deepcopy(self.config), RunnerType.ONNX)
            metric_onnx = evaluator.validate()
            target_metric = evaluator.expected_outputs.get('target_metric')
            self.assertGreaterEqual(metric_onnx, target_metric)

        def test_complete_model_ir_export(self):
            if not self.config.get('export_ir'):
                return
            self.exporter.export_to_onnx_model_if_not_yet(model=self.res_model_name, model_type=None)
            res_model_name_bin = self.res_model_name.replace('onnx', 'bin')
            res_model_name_xml = self.res_model_name.replace('onnx', 'xml')
            self.exporter.export_complete_model_ir()
            result_model_exists = all([os.path.exists(res_model_name_bin), os.path.exists(res_model_name_xml)])
            self.assertEqual(True, result_model_exists)

        def test_run_ir_model(self):
            if not self.config.get('export_ir'):
                return
            self.exporter.export_to_ir_model_if_not_yet(model=self.res_model_name, model_type=None)
            evaluator = Evaluator(deepcopy(self.config), RunnerType.OpenVINO)
            ir_metric = evaluator.validate()
            target_metric = evaluator.expected_outputs.get('target_metric')
            self.assertGreaterEqual(ir_metric, target_metric)

    return TestMonolithicExport


def create_export_test_case_for_composite(config_file, expected_outputs):

    class TestExport(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            export_config = get_config(config_file, section='export')
            export_config['dataset'] = get_config(config_file, section='eval')['dataset']
            cls.config = export_config
            cls.config.update({'expected_outputs': expected_outputs})
            cls.model_path = os.path.join(mkdtemp(), os.path.split(cls.config.get('model_path'))[1])
            cls.encoder_name = os.path.join(os.path.dirname(cls.model_path), cls.config.get('res_encoder_name'))
            cls.decoder_name = os.path.join(os.path.dirname(cls.model_path), cls.config.get('res_decoder_name'))
            cls.config['res_encoder_name'] = cls.encoder_name
            cls.config['res_decoder_name'] = cls.decoder_name
            cls.config['model_path'] = cls.model_path
            if not os.path.exists(cls.model_path):
                download_checkpoint(cls.model_path, cls.config.get('model_url'))
            cls.exporter = Exporter(deepcopy(cls.config))

        def test_encoder_export(self):
            self.exporter.export_encoder()
            result_model_exists = os.path.exists(self.encoder_name)
            self.assertEqual(True, result_model_exists)

        def test_decoder_export(self):
            self.exporter.export_decoder()
            result_model_exists = os.path.exists(self.decoder_name)
            self.assertEqual(True, result_model_exists)

        def test_onnx(self):
            self.exporter.export_to_onnx_model_if_not_yet(model=self.encoder_name, model_type='encoder')
            self.exporter.export_to_onnx_model_if_not_yet(model=self.decoder_name, model_type='decoder')
            evaluator = Evaluator(deepcopy(self.config), RunnerType.ONNX)
            metric_onnx = evaluator.validate()
            target_metric = evaluator.expected_outputs.get('target_metric')
            self.assertGreaterEqual(metric_onnx, target_metric)

        def test_encoder_ir_export(self):
            if not self.config.get('export_ir'):
                return
            self.exporter.export_to_onnx_model_if_not_yet(model=self.encoder_name, model_type='encoder')
            encoder_res_name_bin = self.encoder_name.replace('onnx', 'bin')
            encoder_res_name_xml = self.encoder_name.replace('onnx', 'xml')

            self.exporter.export_encoder_ir()
            result_model_exists = all([os.path.exists(encoder_res_name_bin), os.path.exists(encoder_res_name_xml)])
            self.assertEqual(True, result_model_exists)

        def test_decoder_ir_export(self):
            if not self.config.get('export_ir'):
                return
            self.exporter.export_to_onnx_model_if_not_yet(model=self.decoder_name, model_type='decoder')
            decoder_res_name_xml = self.decoder_name.replace('onnx', 'xml')
            decoder_res_name_bin = self.decoder_name.replace('onnx', 'bin')

            self.exporter.export_decoder_ir()
            result_model_exists = all([os.path.exists(decoder_res_name_xml), os.path.exists(decoder_res_name_bin)])
            self.assertEqual(True, result_model_exists)

        def test_run_ir_model(self):
            if not self.config.get('export_ir'):
                return
            self.exporter.export_to_ir_model_if_not_yet(model=self.encoder_name, model_type='encoder')
            self.exporter.export_to_ir_model_if_not_yet(model=self.decoder_name, model_type='decoder')
            evaluator = Evaluator(deepcopy(self.config), RunnerType.OpenVINO)
            ir_metric = evaluator.validate()
            target_metric = evaluator.expected_outputs.get('target_metric')
            self.assertGreaterEqual(ir_metric, target_metric)

    return TestExport


class TestMediumRenderedExport(
        create_export_test_case_for_composite(
            'configs/medium_config.yml',
            expected_outputs='tests/expected_outputs/formula_recognition/medium_photographed_0185.json')):
    'Test case for medium formula recognition config'


class TestHandwrittenPolynomialsExport(
        create_export_test_case_for_composite(
            'configs/polynomials_handwritten_config.yml',
            expected_outputs='tests/expected_outputs/formula_recognition/polynomials_handwritten_0166.json')):
    'Test case for handwritten polynomials formula recogntion config'


class TestAlphanumeric0014Export(
        create_export_test_case_for_monolithic(
            'configs/config_0014.yml',
            expected_outputs='tests/expected_outputs/alphanumeric/icdar13_greater3_0014.json')):
    'Test case for alphanumeric text recognition config'


class TestAlphanumeric0015Export(
        create_export_test_case_for_composite(
            'configs/config_0015.yml',
            expected_outputs='tests/expected_outputs/alphanumeric/icdar13_greater3_0015.json')):
    'Test case for alphanumeric text recognition config'


class TestAlphanumeric0016Export(
        create_export_test_case_for_composite(
            'configs/config_0016.yml',
            expected_outputs='tests/expected_outputs/alphanumeric/icdar13_greater3_0016.json')):
    'Test case for alphanumeric text recognition config'


if __name__ == '__main__':
    unittest.main()
