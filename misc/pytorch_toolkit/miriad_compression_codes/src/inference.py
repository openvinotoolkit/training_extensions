import argparse
from utils.inference_utils import inference_model


def main(args):
    config = {
        'model_file': args.model_file,
        'inferdata': args.inferdata,
        'bit_depth': args.bit_depth,
        'gpu': args.gpu,
        'with_aac': args.with_aac,
        'produce_latent_code': args.produce_latent_code,
        'produce_decompressed_image': args.produce_decompressed_image,
        'out_latent': args.out_latent,
        'out_decom': args.out_decom,
        'plot_json': args.plot_json,
        'depth': args.depth,
        'width': args.width,
        'max_samples': args.max_samples,
        'phase': args.phase
    }
    inference_model(config)


if __name__ == '__main__':

    parser = argparse.ArgumentParser("""
	The inference script works on a folder level. 
	Provide this script a folder (--inferdata) with full-scale mammograms which can be created with the dataprep.py script.
	It also requires a model (--model_file) file to work on which is to be produced by train.py script.
	All the full-scale mammograms will be processed on by one to collect bpp, ssim and psnr information which will be averaged to produce the numbers to be plotted.
	
	To produce bpp-ssim-psnr with varying bit-depth, run the over and over JUST with different --bit_depth. Keep the json filename same.

	(Optionally) Provide a json filename to write the metrics to.
	(Optionally) You may choose to use AAC (Huffman coding).
	(Optionally) You may choose to use GPU for all processing.
	(Optionally) You may choose to produce (--produce_latent_code and --out_latent) the latent code (integer) and the corresponding huffman codebook.
	(Optionally) You may choose to produce (--produce_decompressed_image and --out_decom) the decompressed images for visualization purpose.
	(Optionally) You may choose the number of Convolution blocks in the architecture. (CVPR paper have 3)
		""")
    parser.add_argument('-m', '--model_file', required=True,
                        type=str, help='Path to the model file')
    parser.add_argument('--inferdata', type=str, required=True,
                        help='Path to the folder containing full scale mammogram images for inference')
    parser.add_argument('-t', '--bit_depth', required=True, type=int,
                        help='Required bit depth for Float2Int quantization')
    parser.add_argument('--gpu', action='store_true', help='Want GPU ?')
    parser.add_argument('--with_aac', action='store_true',
                        help='Use Adaptive Arithmatic Coding (Huffman coding)')
    parser.add_argument('-l', '--produce_latent_code', action='store_true',
                        help='Write latent code tensor (integer) as output (possibly along huffman codebook)')
    parser.add_argument('-d', '--produce_decompressed_image',
                        action='store_true', help='Write decompressed images as output')
    parser.add_argument('--out_latent', type=str, required=False,
                        default='.', help='Folder to produce the latent codes into ?')
    parser.add_argument('--out_decom', type=str, required=False,
                        default='.', help='Folder to produce decompressed images into ?')
    parser.add_argument('-p', '--plot_json', type=str, required=False,
                        default='./plot.json', help='Path for the output json file')
    parser.add_argument('--depth', type=int, required=False,
                        default=3, help='no of conv blocks in the architecture')
    parser.add_argument('--width', type=int, required=False,
                        default=64, help='no of Filters')
    parser.add_argument('-x', '--max_samples', type=int, required=False,
                        default=0, help='limit the number of samples to use for inference')
    parser.add_argument('-ph', '--phase', type=int,
                        required=True, default=1, help='Phase')
    args = parser.parse_args()

    main(args)
