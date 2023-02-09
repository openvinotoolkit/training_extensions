import argparse
from utils.inference_utils import inference_model

def main(args):
    config = {
        'model_file': args.model_file,
        'data': args.dataset,
        'gpu': args.gpu,
        'backbone': args.backbone,
        'split_npz': args.split_npz,
        'max_samples': args.max_samples,
        'gnn': args.gnn,
        'checkpoint': args.checkpoint
    }
    inference_model(config,run_type=args.run_type)


if __name__ == '__main__':

    parser = argparse.ArgumentParser("""
    The inference script works on a folder level. 
    Provide this script a folder (--data) with full-scale mammograms which can be created with the dataprep.py script.
    It also requires a model (--model_file) file to work on which is to be produced by train.py script.
    All the full-scale mammograms will be processed on by one to collect bpp, ssim and psnr information which will be averaged to produce the numbers to be plotted.
    """)
    parser.add_argument('--data', type=str, required=True,
                        help='Folder path of dataset')
    parser.add_argument('--split_npz', type=str, required=True,
                        help='Folder path to the full-scale mammogram images (test)')
    parser.add_argument('--gpu', help='Want GPU ?',type=str)
    parser.add_argument('--model_file', type=str,
                        required=True, help='Name of model and log files')
    parser.add_argument('--max_samples', type=int, help='no. of max samples to infer on', default=10)
    parser.add_argument('--gnn', action='store_true',type=str , help='using gnn or not?')
    parser.add_argument('--backbone', type=str, required=False, default='resnet', help='backbone model')
    parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint to refer for onnx and ir models')
    arguments = parser.parse_args()

    main(arguments)
