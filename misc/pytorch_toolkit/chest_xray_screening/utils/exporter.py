import torch
import argparse
import torch.nn as nn
import os
import subprocess
from model import DenseNet121,DenseNet121Eff


OPENVINO_DIR = '/opt/intel/openvino_2021'

class Exporter:
    def __init__(self, config):

        self.config = config
        if self.config['optimised'] == True:
            self.model = DenseNet121Eff(self.config["alpha"], self.config["beta"], self.config["class_count"])
        else:
            self.model = DenseNet121(class_count=3)

        self.model.eval()
        self.model_path = config.get('checkpoint')
        if self.model_path is not None:
            self.model.load_weights(self.model_path)

    def export_model_ir(self):
            input_model = os.path.join(os.path.split(self.model_path)[0], self.config.get('model_name'))
            input_shape = self.config.get('input_shape')
            output_dir = os.path.split(self.model_path)[0]
            export_command = f"""{OPENVINO_DIR}/bin/setupvars.sh && \
            python {OPENVINO_DIR}/deployment_tools/model_optimizer/mo.py \
            --framework onnx \
            --input_model {input_model} \
            --input_shape "{input_shape}" \
            --output_dir {output_dir} \
            --scale_values 'imgs[255]'"""
            if self.config.get('verbose_export'):
                print(export_command)
            subprocess.run(export_command, shell=True, check=True)

    def export_model_onnx(self):
            print(f"Saving model to {self.config.get('model_name')}")
            res_path = os.path.join(os.path.split(self.model_path)[0], self.config.get('model_name'))
            dummy_input = torch.randn(1, 3, 1024, 1024)
            torch.onnx.export(self.model, dummy_input, res_path,
                            opset_version=11, verbose=False)

