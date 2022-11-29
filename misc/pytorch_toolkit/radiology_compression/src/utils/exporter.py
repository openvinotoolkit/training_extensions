import torch
import os
import subprocess
from .model import load_checkpoint
from .train_utils import load_model

class Exporter:
    def __init__(self, config, phase):
        self.config = config
        self.checkpoint = config.get('checkpoint')
        self.phase = phase
        if phase == 1:
            alpha = self.config['alpha'] ** self.config['phi']
            beta = self.config['beta'] ** self.config['phi']
            self.model = load_model(
                eff_flag=False, it_no=0, alpha=alpha, beta=beta,
                depth=3, width=64, phase=1, phi=self.config['phi'])
        else:
            self.model = load_model(
                eff_flag=False, phase=2, phi=self.config['phi'])

        self.model.eval()
        load_checkpoint(self.model, self.checkpoint)

    def export_model_ir(self):
        input_model = os.path.join(
            os.path.split(self.checkpoint)[0], self.config.get('model_name_onnx'))
        input_shape = self.config.get('input_shape')
        output_dir = os.path.split(self.checkpoint)[0]
        export_command = f"""mo \
        --framework onnx \
        --input_model {input_model} \
        --input_shape "{input_shape}" \
        --output_dir {output_dir}"""

        if self.config.get('verbose_export'):
            print(export_command)
        subprocess.run(export_command, shell=True, check=True)

    def export_model_onnx(self):

        print(f"Saving model to {self.config.get('model_name_onnx')}")
        res_path = os.path.join(os.path.split(self.checkpoint)[0], self.config.get('model_name_onnx'))

        if self.phase == 2:
            dummy_input = torch.randn(1, 16, 160, 324)
        else:
            dummy_input = torch.randn(1, 1, 1024, 1024)

        torch.onnx.export(self.model, dummy_input, res_path,
                          opset_version=11, do_constant_folding=True,
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}},
                          verbose=False)
