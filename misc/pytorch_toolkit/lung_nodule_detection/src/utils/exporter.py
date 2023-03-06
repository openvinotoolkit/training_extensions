import torch
import os
from .utils import load_model
from .utils import load_checkpoint

class Exporter:
    def __init__(self, config, stage):
        self.config = config
        self.checkpoint = config.get('checkpoint')
        self.stage = stage
        self.model = load_model(network=config["network"])
        self.model.eval()
        load_checkpoint(self.model, self.checkpoint)

    def export_model_ir(self):
        input_model = os.path.join(
            os.path.split(self.checkpoint)[0], self.config.get('model_name_onnx'))
        input_shape = self.config.get('input_shape')
        output_dir = os.path.split(self.checkpoint)[0]
        current_dir =  os.path.abspath(os.path.dirname(__file__))
        openvino_extension_path = os.path.join(current_dir,'openvino_pytorch_layers','mo_extensions')
        export_command = f"""mo \
        --framework onnx \
        --input_model {input_model} \
        --input_shape "{input_shape}" \
        --output_dir {output_dir}\
        --log_level=ERROR\
        --extension {openvino_extension_path}"""

        if self.config.get('verbose_export'):
            print(export_command)
        os.system(export_command)

    def export_model_onnx(self):

        print(f"Saving model to {self.config.get('model_name_onnx')}")
        res_path = os.path.join(os.path.split(self.checkpoint)[0], self.config.get('model_name_onnx'))

        if self.stage == 2:
            dummy_input = torch.randn(1, 1, 64, 64)
        else:
            dummy_input = torch.randn(1, 1, 512, 512)

        torch.onnx.export(self.model, dummy_input, res_path,
                          input_names=['input'], output_names=['output'],
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
                          verbose=False)
