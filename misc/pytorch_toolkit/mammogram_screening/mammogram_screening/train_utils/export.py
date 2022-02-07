import torch
import os
from .models import UNet, Model2
import subprocess
from .get_config import get_config


class Exporter:
    def __init__(self, config, stage):

        self.config = config
        if stage == 'stage1':
            self.model = UNet(num_filters=32)
        else:
            self.model = Model2()

        checkpoint = torch.load(self.config['checkpoint'], map_location=torch.device('cuda'))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to('cuda')
        self.model.eval()

    def export_model_ir(self, stage):
        input_model = self.config['onnx_model']
        if stage == 'stage1':
            input_shape = "[1,1,640,320]"
        else:
            input_shape = "[5,1,64,64]"
        # input_shape = self.config['input_shape']
        output_dir = os.path.split(self.config['checkpoint'])[0]
        export_command = f"""mo \
        --framework onnx \
        --input_model {input_model} \
        --input_shape "{input_shape}" \
        --output_dir {output_dir} \
        --log_level DEBUG"""

        subprocess.run(export_command, shell = True, check = True)

    def export_model_onnx(self, stage):
        print(f"Saving model to {self.config['onnx_model']}")
        res_path = self.config['onnx_model']
        if stage == 'stage1':
            dummy_input = torch.randn(1, 1, 640, 320).cuda()
        else:
            dummy_input = torch.randn(5, 1, 64, 64).cuda()
        torch.onnx.export(self.model, dummy_input, res_path,
                        opset_version = 11, do_constant_folding=True,
                        input_names = ['input'], output_names=['output'],
                        dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}},
                        verbose = False)

if __name__ == '__main__':
    
    config = get_config(action='export', stage='stage1')
    exporter = Exporter(config=config, stage='stage1')
    exporter.export_model_onnx(stage='stage1')
    exporter.export_model_ir(stage='stage1')

    config = get_config(action='export', stage='stage2')
    exporter = Exporter(config=config, stage='stage2')
    exporter.export_model_onnx(stage='stage2')
    exporter.export_model_ir(stage='stage2')
