import torch
import os
from .model import Infer_model
from .misc import aggregate_local_weights
class Exporter:
    def __init__(self, config, gnn=True):
        self.config = config
        self.checkpoint = config.get('checkpoint')
        self.gnn = gnn
        if self.gnn:
            model = Infer_model(config.get('backbone'),config.get('split_path'),gnn=True)
        else:
            model = Infer_model(config.get('backbone'),config.get('split_path'),gnn=False)
        self.model = model
        self.model.eval()
        checkpoint=torch.load(self.checkpoint)
        glbl_cnv_wt=checkpoint['cnv_lyr_state_dict']
        glbl_backbone_wt=checkpoint['backbone_model_state_dict']
        glbl_fc_wt=checkpoint['fc_layers_state_dict']

        self.model.cnv_lyr.load_state_dict(glbl_cnv_wt)
        self.model.backbone_model.load_state_dict(glbl_backbone_wt)
        self.model.fc_layers.load_state_dict(glbl_fc_wt)
        if self.gnn:

            sit0_gnn_wt=checkpoint['sit0_gnn_model']
            sit1_gnn_wt=checkpoint['sit1_gnn_model']
            sit2_gnn_wt=checkpoint['sit2_gnn_model']
            sit3_gnn_wt=checkpoint['sit3_gnn_model']
            sit4_gnn_wt=checkpoint['sit4_gnn_model']
            glbl_gnn_wt=aggregate_local_weights(sit0_gnn_wt, sit1_gnn_wt,
                                                sit2_gnn_wt,sit3_gnn_wt,
                                                sit4_gnn_wt, torch.device('cpu'))
            self.model.gnn_model.load_state_dict(glbl_gnn_wt)

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
        os.system(export_command)

    def export_model_onnx(self):

        print(f"Saving model to {self.config.get('model_name_onnx')}")
        res_path = os.path.join(os.path.split(self.checkpoint)[0], self.config.get('model_name_onnx'))

        dummy_input = torch.randn(1, 1, 320, 320)

        torch.onnx.export(self.model, dummy_input, res_path,
                        opset_version=11, do_constant_folding=True,
                        input_names=['input'], output_names=['output'],
                        dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}},
                        verbose=True)
