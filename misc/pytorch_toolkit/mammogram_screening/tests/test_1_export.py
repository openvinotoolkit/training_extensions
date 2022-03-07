import unittest
import os
from mammogram_screening.train_utils.get_config import get_config
from mammogram_screening.train_utils.export import Exporter
from mammogram_screening.train_utils.downloader import download_checkpoint

def create_export_test_for_stage1():
    class ExportTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.config = get_config(action = 'export', config_path='configs/', stage='stage1')
            if not os.path.exists(cls.config['checkpoint']):
                download_checkpoint()

        def test_export_onnx(self):
            self.exporter = Exporter(self.config, stage='stage1')
            self.exporter.export_model_onnx(stage='stage1')
            flag = os.path.exists(self.config['onnx_model'])
            self.assertTrue(flag)

        def test_export_ir(self):
            self.exporter = Exporter(self.config, stage='stage1')
            model_dir = os.path.split(self.config['checkpoint'])[0]
            if not os.path.exists(self.config['onnx_model']):
                self.exporter.export_model_onnx(stage='stage1')
            self.exporter.export_model_ir(stage='stage1')
            name_xml = self.config['model_name'] + '.xml'
            name_bin = self.config['model_name'] + '.bin'
            xml_status = os.path.exists(os.path.join(model_dir, name_xml))
            bin_status = os.path.exists(os.path.join(model_dir, name_bin))
            self.assertTrue(xml_status)
            self.assertTrue(bin_status)

    return ExportTest

def create_export_test_for_stage2():
    class ExportTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.config = get_config(action = 'export', config_path='configs/', stage='stage2')
            if not os.path.exists(cls.config['checkpoint']):
                download_checkpoint()

        def test_export_onnx(self):
            self.exporter = Exporter(self.config, stage='stage2')
            self.exporter.export_model_onnx(stage='stage2')
            flag = os.path.exists(self.config['onnx_model'])
            self.assertTrue(flag)

        def test_export_ir(self):
            self.exporter = Exporter(self.config, stage='stage2')
            model_dir = os.path.split(self.config['checkpoint'])[0]
            if not os.path.exists(self.config['onnx_model']):
                self.exporter.export_model_onnx(stage='stage2')
            self.exporter.export_model_ir(stage='stage2')
            name_xml = self.config['model_name'] + '.xml'
            name_bin = self.config['model_name'] + '.bin'
            xml_status = os.path.exists(os.path.join(model_dir, name_xml))
            bin_status = os.path.exists(os.path.join(model_dir, name_bin))
            self.assertTrue(xml_status)
            self.assertTrue(bin_status)

    return ExportTest

class TestInferenceStage1(create_export_test_for_stage1()):
    'Test case for Stage1'

class TestInferenceStage2(create_export_test_for_stage2()):
    'Test case for Stage2'

if __name__ == '__main__':

    unittest.main()
