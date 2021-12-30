import unittest
import os
from src.exporter import export_model_onnx, export_model_ir
from src.utils.read_config import load_config

class ExportTest(unittest.TestCase):
        
    def test_export_onnx(self):
        net_config_path = "configs/net_config.json"
        net_config = load_config(net_config_path)
        export_config = net_config['export']
        export_model_onnx(export_config)
        self.model_path = export_config['model_path']
        self.assertTrue(os.path.join(os.path.split(self.model_path)[0], export_config['onnx_model_name']))

    def test_export_ir(self):
        net_config_path = "configs/net_config.json"
        net_config = load_config(net_config_path)
        export_config = net_config['export']
        export_model_ir(export_config)
        self.model_path = export_config['model_path']
        self.assertTrue(os.path.join(os.path.split(self.model_path)[0], export_config['ir_model_name']))

if __name__ == '__main__':

    unittest.main()