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

from im2latex.utils.exporter import Exporter
from im2latex.utils.evaluator import Evaluator, RunnerType
from im2latex.utils.get_config import get_config


def create_export_test_case(config_file, expected_outputs):
    class TestExport(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            export_config = get_config(config_file, section='export')
            val_config = get_config(config_file, section='eval')
            export_config['test_path'] = val_config['test_path']
            cls.config = export_config
            cls.config.update({"expected_outputs": expected_outputs})
            cls.exporter = Exporter(cls.config)
            print("test case for config {} created".format(config_file))

        def test_1_encoder_export(self):
            encoder_res_name = self.config.get("res_encoder_name")
            result_model_exists = os.path.exists(encoder_res_name)
            if result_model_exists:
                os.remove(encoder_res_name)
            self.exporter.export_encoder()
            result_model_exists = os.path.exists(encoder_res_name)
            self.assertEqual(True, result_model_exists)

        def test_2_decoder_export(self):
            decoder_res_name = self.config.get("res_decoder_name")
            result_model_exists = os.path.exists(decoder_res_name)
            if result_model_exists:
                os.remove(decoder_res_name)
            self.exporter.export_decoder()
            result_model_exists = os.path.exists(decoder_res_name)
            self.assertEqual(True, result_model_exists)

        def test_3_onnx(self):
            for model_type in ('encoder', 'decoder'):
                self.exporter.export_to_onnx_model_if_not_yet(model=self.config.get(
                    f"res_{model_type}_name"), model_type=model_type)
            evaluator = Evaluator(self.config, RunnerType.ONNX)
            metric_onnx = evaluator.validate()
            target_metric = evaluator.expected_outputs.get("target_metric")
            self.assertGreaterEqual(metric_onnx, target_metric)

        def test_4_encoder_ir_export(self):
            if not self.config.get("export_ir"):
                return
            self.exporter.export_to_onnx_model_if_not_yet(model=self.config.get(
                "res_encoder_name"), model_type='encoder')
            encoder_res_name = self.config.get("res_encoder_name").replace('onnx', 'xml')
            result_model_exists = os.path.exists(encoder_res_name)
            if result_model_exists:
                os.remove(encoder_res_name)
            encoder_res_name = self.config.get("res_encoder_name").replace('onnx', 'bin')
            result_model_exists = os.path.exists(encoder_res_name)
            if result_model_exists:
                os.remove(encoder_res_name)

            self.exporter.export_encoder_ir()
            encoder_res_name = self.config.get("res_encoder_name").replace('onnx', 'bin')
            result_model_exists = os.path.exists(encoder_res_name)
            self.assertEqual(True, result_model_exists)
            encoder_res_name = self.config.get("res_encoder_name").replace('onnx', 'xml')
            result_model_exists = os.path.exists(encoder_res_name)
            self.assertEqual(True, result_model_exists)

        def test_5_decoder_ir_export(self):
            if not self.config.get("export_ir"):
                return
            self.exporter.export_to_onnx_model_if_not_yet(model=self.config.get(
                "res_decoder_name"), model_type='decoder')
            decoder_res_name = self.config.get("res_decoder_name").replace('onnx', 'xml')
            result_model_exists = os.path.exists(decoder_res_name)
            if result_model_exists:
                os.remove(decoder_res_name)
            decoder_res_name = self.config.get("res_decoder_name").replace('onnx', 'bin')
            result_model_exists = os.path.exists(decoder_res_name)
            if result_model_exists:
                os.remove(decoder_res_name)

            self.exporter.export_decoder_ir()
            decoder_res_name = self.config.get("res_decoder_name").replace('onnx', 'bin')
            result_model_exists = os.path.exists(decoder_res_name)
            self.assertEqual(True, result_model_exists)
            decoder_res_name = self.config.get("res_decoder_name").replace('onnx', 'xml')
            result_model_exists = os.path.exists(decoder_res_name)
            self.assertEqual(True, result_model_exists)

        def test_6_run_ir_model(self):
            if not self.config.get("export_ir"):
                return
            for model_type in ('encoder', 'decoder'):
                self.exporter.export_to_ir_model_if_not_yet(model=self.config.get(
                    f"res_{model_type}_name"), model_type=model_type)
            evaluator = Evaluator(self.config, RunnerType.OpenVINO)
            ir_metric = evaluator.validate()
            target_metric = evaluator.expected_outputs.get("target_metric")
            self.assertGreaterEqual(ir_metric, target_metric)
    return TestExport


class TestMediumRenderedExport(
        create_export_test_case(
            "configs/medium_config.yml",
            expected_outputs="tests/expected_outputs/formula_recognition/medium_photographed_0185.json")):
    "Test case for medium config"


class TestHandwrittenPolynomialsExport(
        create_export_test_case(
            'configs/polynomials_handwritten_config.yml',
            expected_outputs='tests/expected_outputs/formula_recognition/polynomials_handwritten_0166.json')):
    "Test case for handwritten polynomials config"


if __name__ == "__main__":
    unittest.main()
