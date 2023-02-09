import unittest
import os
import sys
sys.path.append(r'/storage/adityak/federated/')
from src.utils.downloader import download_checkpoint
from src.utils.exporter import Exporter
from src.utils.get_config import get_config

def create_export_test_with_gnn():
    class ExportTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.config = get_config(action='export',gnn=True) 
            if not os.path.exists(cls.config['checkpoint']):
                download_checkpoint(gnn=True)
            cls.model_path = cls.config['checkpoint']

        def test_export_onnx(self):
            self.exporter = Exporter(self.config, gnn=True)
            self.exporter.export_model_onnx()
            self.assertTrue(os.path.join(os.path.split(self.model_path)[
                            0], self.config.get('model_name_onnx')))

        def test_export_ir(self):
            self.exporter = Exporter(self.config, gnn=True)
            model_dir = os.path.split(self.config['checkpoint'])[0]
            if not os.path.exists(os.path.join(model_dir, self.config.get('model_name_onnx'))):
                self.exporter.export_model_onnx()
            self.exporter.export_model_ir()
            name_xml = self.config['model_name'] + '.xml'
            name_bin = self.config['model_name'] + '.bin'
            xml_status = os.path.exists(os.path.join(model_dir, name_xml))
            bin_status = os.path.exists(os.path.join(model_dir, name_bin))
            self.assertTrue(xml_status)
            self.assertTrue(bin_status)

        def test_config(self):
            self.config = get_config(action='export', gnn=True)
            self.model_path = self.config['checkpoint']
            self.input_shape = self.config['input_shape']
            self.output_dir = os.path.split(self.model_path)[0]
            self.assertTrue(self.output_dir)
            self.assertTrue(self.model_path)
            self.assertListEqual(self.input_shape, [1, 1, 320, 320])
    return ExportTest

def create_export_test_without_gnn():
    class ExportTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.config = get_config(action='export',gnn=False) 
            if not os.path.exists(cls.config['checkpoint']):
                download_checkpoint(gnn=False)
            cls.model_path = cls.config['checkpoint']

        def test_export_onnx(self):
            self.exporter = Exporter(self.config, gnn=False)
            self.exporter.export_model_onnx()
            self.assertTrue(os.path.join(os.path.split(self.model_path)[
                            0], self.config.get('model_name_onnx')))

        def test_export_ir(self):
            self.exporter = Exporter(self.config, gnn=False)
            model_dir = os.path.split(self.config['checkpoint'])[0]
            if not os.path.exists(os.path.join(model_dir, self.config.get('model_name_onnx'))):
                self.exporter.export_model_onnx()
            self.exporter.export_model_ir()
            name_xml = self.config['model_name'] + '.xml'
            name_bin = self.config['model_name'] + '.bin'
            xml_status = os.path.exists(os.path.join(model_dir, name_xml))
            bin_status = os.path.exists(os.path.join(model_dir, name_bin))
            self.assertTrue(xml_status)
            self.assertTrue(bin_status)

        def test_config(self):
            self.config = get_config(action='export', gnn=False)
            self.model_path = self.config['checkpoint']
            self.input_shape = self.config['input_shape']
            self.output_dir = os.path.split(self.model_path)[0]
            self.assertTrue(self.output_dir)
            self.assertTrue(self.model_path)
            self.assertListEqual(self.input_shape, [1, 1, 320, 320])
    return ExportTest



class TestInferenceEff(create_export_test_without_gnn()):
    'Test case with gnn'

class TestInference(create_export_test_with_gnn()):
    'Test case with gnn'

if __name__ == '__main__':
    unittest.main()