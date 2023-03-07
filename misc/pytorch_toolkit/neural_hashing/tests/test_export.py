import unittest
import os
from src.utils.exporter import Exporter
from src.utils.get_config import get_config
from src.utils.downloader import download_checkpoint

def create_export_test_for_encoder():
    class ExportTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.config = get_config(action = 'export')
            current_dir =  os.path.abspath(os.path.dirname(__file__))
            parent_dir = os.path.abspath(current_dir + "/../")
            model_dir = parent_dir + os.path.split(cls.config['checkpoint'])[0]
            model_dir = parent_dir + '/src/utils/model_weights/'
            if not os.path.exists(model_dir):
                download_checkpoint()
            cls.model_path = cls.config['checkpoint']

        def test_export_onnx(self):
            self.exporter = Exporter(self.config, openvino = 1)
            current_dir =  os.path.abspath(os.path.dirname(__file__))
            parent_dir = os.path.abspath(current_dir + "/../")
            self.exporter.export_model_onnx(parent_dir)
            self.assertTrue(os.path.join(os.path.split(self.model_path)[0],
                                        parent_dir + self.config.get('model_name_onnx')))


        def test_export_ir(self):
            self.exporter = Exporter(self.config, openvino = 1)
            current_dir =  os.path.abspath(os.path.dirname(__file__))
            parent_dir = os.path.abspath(current_dir + "/../")
            model_dir = parent_dir + os.path.split(self.config['checkpoint'])[0]
            if not os.path.exists(os.path.join(model_dir, self.config.get('model_name_onnx'))):
                self.exporter.export_model_onnx(parent_dir)
            self.exporter.export_model_ir(parent_dir)
            name_xml = self.config['model_name'] + '.xml'
            name_bin = self.config['model_name'] + '.bin'
            xml_status = os.path.exists(os.path.join(model_dir, name_xml))
            bin_status = os.path.exists(os.path.join(model_dir, name_bin))
            self.assertTrue(xml_status)
            self.assertTrue(bin_status)


        def test_config(self):
            self.config = get_config(action = 'export')
            self.model_path = self.config['checkpoint']
            self.input_shape = self.config['input_shape']
            self.output_dir = os.path.split(self.model_path)[0]
            self.assertTrue(self.output_dir)
            self.assertTrue(self.model_path)
            self.assertListEqual(self.input_shape, [1,1, 28, 28])
    return ExportTest

class TestExporter(create_export_test_for_encoder()):
    'Test case for exporting encoder'

if __name__ == '__main__':
    unittest.main()
