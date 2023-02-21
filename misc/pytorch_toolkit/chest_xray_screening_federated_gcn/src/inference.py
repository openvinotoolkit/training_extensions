import argparse
from utils.inference_utils import inference_model

def main(args):
    config = {
        'data': args.data,
        'split_npz': args.split_npz,
        'gpu': args.gpu,
        'gnn': args.gnn,
        'backbone': args.backbone,
        'checkpoint': args.checkpoint,
        'max_samples': args.max_samples,
        'model_file': args.model_file
    }
    inference_model(config,run_type=args.run_type)


if __name__ == '__main__':

    parser = argparse.ArgumentParser("""
    Inference script.
    Provide the folder paths which contains dataset.
    Provide the folder paths which contains split.npz file.
    (Optionally) You may choose to use GPU (--gpu).
    (Optionally) You may choose to use GNN (--gnn).
    (optionally) You may choose to use resnet/densenet/xception for backbone. Default is resnet (--backbone).
    (Optionally) You may choose to provide path to model weights, to run inference using pytorch(--model_file).
    (Optionally) You may choose to provide path to onnx or ir model, for running inference if run_type is onnx or ir(--checkpoint).
    (Optionally) You may choose to provide the max no. of samples to run inference on(--max_samples). Default value is 10.
    (Optionally) You may choose to provide run type(--run_type). Default runtype is pytorch.
    """)
    parser.add_argument('--data', type=str, required=True,
                        help='Folder path of dataset')
    parser.add_argument('--split_npz', type=str, required=True,
                        help='path to the split.npz file')
    parser.add_argument('--gpu', type=str, help='Want GPU ?', required=False, default='False')
    parser.add_argument('--model_file', type=str, required=False,
                        help='Path of model weights saved for running inference with pytorch')
    parser.add_argument('--checkpoint', type=str,required=False,
                        help='Path of onnx model file to load for inference. Required if run type is onnx or ir')
    parser.add_argument('--gnn', type=str, help='using gnn or not?',
                        required=False, default='False')
    parser.add_argument('--backbone', type=str, required=False, default='resnet',
                        help='backbone model to be used from resnet/densenet/xception, by default resnet is used')
    parser.add_argument('--max_samples', type=int, required=False, default=10,
                        help='Max no. of samples to run inference on' )
    parser.add_argument('--run_type', type=str, required=False, default='pytorch',
                        help='Chosse run type out of pytorch, onnx, ir. Default run type is pytorch')
    arguments = parser.parse_args()

    main(arguments)
