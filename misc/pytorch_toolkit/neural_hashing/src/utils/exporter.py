import torch
import os
from src.utils.network import Encoder, load_checkpoint


class Exporter:
    def __init__(self, config, openvino=0):

        self.config = config
        self.checkpoint = config.get('checkpoint')
        self.model = Encoder(self.config['zSize'])
        if openvino == 0:
            load_checkpoint(self.model, self.checkpoint)
        else:
            current_dir =  os.path.abspath(os.path.dirname(__file__))
            parent_dir = os.path.abspath(current_dir + "/../../")
            load_checkpoint(self.model, parent_dir + self.checkpoint)

    def export_model_ir(self, parent_dir):
        input_model = parent_dir + self.config.get('model_name_onnx')
        input_shape = self.config.get('input_shape')
        output_dir = parent_dir + "/src/utils/model_weights"
        export_command = f"""mo \
        --framework onnx \
        --input_model {input_model} \
        --input_shape "{input_shape}" \
        --output_dir {output_dir}"""

        if self.config.get('verbose_export'):
            print(export_command)

        os.system(export_command)

    def export_model_onnx(self, parent_dir):
        print(f"Saving model to {parent_dir + self.config.get('model_name_onnx')}")
        res_path = os.path.join(os.path.split(self.checkpoint)[0], parent_dir + self.config.get('model_name_onnx'))
        dummy_input = torch.randn(1,1, 28, 28)
        torch.onnx.export(self.model, dummy_input, res_path,
                        opset_version = 11, do_constant_folding=True,
                        input_names = ['input'], output_names=['output'],
                        dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}},
                        verbose = False)
