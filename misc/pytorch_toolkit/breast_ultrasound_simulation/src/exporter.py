import torch
from .model import GeneratorModel
import argparse
import os
import subprocess

# OPENVINO_DIR = '/opt/intel/openvino_2021'

def export_model_onnx(config_args):
    # if config_args.gpu:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available())
    model_path = config_args['model_path']
    gen_old = GeneratorModel(1).to(device)
    state_dict = torch.load(model_path)
    gen_old.load_state_dict(state_dict["generator1_weights"])
    onnx_model = config_args['onnx_model_name']
    dummy_input = torch.randn(1,1,128,128).cuda()
    # torch.onnx.export(gen_old, dummy_input, onnx_model, verbose=True)
    torch.onnx.export(gen_old, dummy_input, onnx_model,
                        opset_version = 11, do_constant_folding=True,
                        input_names = ['input'], output_names=['output'],
                        dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}},
                        verbose = False)

def export_model_nx_ir(config_args):
    input_model = config_args['onnx_model_name']
    input_shape = [1,1,128,128]
    output_dir = os.path.split(config_args['ir_model_name'])[0]
    export_command = f"""mo \
        --framework onnx \
        --input_model {input_model} \
        --input_shape "{input_shape}" \
        --output_dir {output_dir}"""
    subprocess.run(export_command, shell = True, check = True)

def main(config_args):
    if config_args.onnx:
        export_model_onnx(config_args)
    if config_args.ir:
        export_model_ir(config_args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, help='absolute path to model weights', type=str)
    parser.add_argument('--name', required=True, help='Model name with .onnx extension', type=str)

    args = parser.parse_args()
    main(args)
