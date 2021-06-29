import torch
from model import GeneratorModel
import argparse

OPENVINO_DIR = '/opt/intel/openvino_2021'

def export_model_onnx(args):
    if args.gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(torch.cuda.is_available())

    model_path = args.path

    gen_old = GeneratorModel(1).to(device)

    state_dict = torch.load(model_path)
    gen_old.load_state_dict(state_dict["generator1_weights"])
    onnx_model = args.name
    dummy_input = torch.randn(1,1,128,128).cuda()
    torch.onnx.export(gen_old, dummy_input, onnx_model, verbose=True)

def export_model_ir(args):
    input_model = args.path
    input_shape = [1,1,128,128]
    output_dir = os.path.split(args.path)[0]
    export_command = f"""{OPENVINO_DIR}/bin/setupvars.sh && \
    python {OPENVINO_DIR}/deployment_tools/model_optimizer/mo.py \
    --framework onnx \
    --input_model {input_model} \
    --input_shape "{input_shape}" \
    --output_dir {output_dir} """
    if self.config.get('verbose_export'):
        print(export_command)
    subprocess.run(export_command, shell=True, check=True)

def main(args):
    if args.onnx:
        export_model_onnx(args)
    if args.ir:
        export_model_ir(args)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, help='absolute path to model weights', type=str)
    parser.add_argument('--gpu', required=False, help='To set device as GPU', type=bool)
    parser.add_argument('--name', required=True, help='Model name with .onnx extension', type=str)

    args = parser.parse_args()
    main(args)

