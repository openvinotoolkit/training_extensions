import torch
import os
import subprocess
from src.utils.predict_utils import load_volumetric_model


class Exporter:
    def __init__(self, config):
        self.config = config
        self.model = load_volumetric_model(
            model_name=config["model_name"], 
            model_filename=config["model_filename"], 
            n_outputs=config["n_outputs"], 
            n_features=config["n_features"], 
            n_gpus=1, 
            strict=True, window=config["window"])

    def export_model_ir(self):
        input_model = self.config["model_onnx_filename"]
        input_shape = [2, 1, 128, 160, 160]
        export_command = f"""mo \
        --framework onnx \
        --input_model {input_model} \
        --input_shape "{input_shape}" \
        --log_level DEBUG"""

        if self.config.get('verbose_export'):
            print(export_command)
        subprocess.run(export_command, shell = True, check = True)

    def export_model_onnx(self):
        print(f"Saving model to {self.config['model_onnx_filename']}")
        res_path = self.config['model_onnx_filename']
        dummy_input = torch.randn(2, 1, 128, 160, 160).cuda()
        torch.onnx.export(self.model, dummy_input, res_path,
                        opset_version = 11, do_constant_folding=True,
                        input_names = ['input'], output_names=['output'],
                        verbose = False)
