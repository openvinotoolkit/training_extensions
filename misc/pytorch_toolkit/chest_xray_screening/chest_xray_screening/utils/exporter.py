import torch
import os
import subprocess
from .model import DenseNet121, DenseNet121Eff, load_checkpoint



class Exporter:
    def __init__(self, config, optimised):

        self.config = config
        self.checkpoint = config.get('checkpoint')
        if optimised:
            alpha =  self.config['alpha'] ** self.config['phi']
            beta = self.config['beta'] ** self.config['phi']
            self.model = DenseNet121Eff(alpha, beta, self.config["class_count"])
        else:
            self.model = DenseNet121(class_count = 3)

        self.model.eval()
        load_checkpoint(self.model, self.checkpoint)

    def export_model_ir(self):
        input_model = os.path.join(os.path.split(self.checkpoint)[0], self.config.get('model_name_onnx'))
        input_shape = self.config.get('input_shape')
        output_dir = os.path.split(self.checkpoint)[0]
        export_command = f"""mo \
        --framework onnx \
        --input_model {input_model} \
        --input_shape "{input_shape}" \
        --output_dir {output_dir}"""

        if self.config.get('verbose_export'):
            print(export_command)
        subprocess.run(export_command, shell = True, check = True)

    def export_model_onnx(self):
        print(f"Saving model to {self.config.get('model_name_onnx')}")
        res_path = os.path.join(os.path.split(self.checkpoint)[0], self.config.get('model_name_onnx'))
        dummy_input = torch.randn(1, 3, 1024, 1024)
        torch.onnx.export(self.model, dummy_input, res_path,
                        opset_version = 11, do_constant_folding=True,
                        input_names = ['input'], output_names=['output'],
                        dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}},
                        verbose = False)
