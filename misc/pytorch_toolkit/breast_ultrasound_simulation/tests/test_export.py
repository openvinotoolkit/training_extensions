import unittest
import os
import json
from src import exporter

class ExportTest(unittest.TestCase):
        
    def test_export_onnx(self):
        self.config = get_config()
        export_model_onnx(self.config)
        self.model_path = self.config['path']
        self.assertTrue(os.path.join(os.path.split(self.model_path)[0], self.config.get('name')))

    def test_export_ir(self):
        self.assertTrue(os.isdir('/opt/intel/openvino_2021'))
        self.config = get_config()
        export_model_ir(self.config)
        self.model_path = self.config['path']
        self.assertTrue(os.path.join(os.path.split(self.model_path)[0], self.config.get('name')))

if __name__ == '__main__':

    unittest.main()