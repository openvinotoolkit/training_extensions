import argparse
from utils.train_utils import train_model

def main(args):
    config = {
        'traindata': args.traindata,
        'testdata': args.testdata,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'gpu': args.gpu,
        'interval': args.interval,
        'subset_size': args.subset_size,
        'model_file_name': args.model_file_name,
        'randomize_subset': args.randomize_subset,
        'depth': args.depth,
        'width': args.width,
        'phi': args.phi,
        'efficient_net': args.efficient_net,
        'iterate': args.iterate,
        'phase': args.phase
    }
    train_model(config)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        """
    Training script.

    Provide the folder paths which contains traning (patch) images and testing (full-scale mammogram) images produced by dataprep.py.
    Provide a filename prefix to help differentiate model/log from different runs with different configurations.

    (Optionally) You may choose to provide batch size (--batch_size), number of epochs (--epoch), printing interval (--interval).
    (Optionally) You may choose to limit the number of samples (--subset_size) to be used per epoch. '0' means the whole dataset.
    (Optionally) You may also choose to randomize (--randomize_subset) the selected subset every epoch. It has effect only if --subset_size is non-zero.
    (Optionally) You may choose to use GPU (--gpu).
    (Optionally) You may choose the number of Convolution blocks in the architecture. (CVPR paper have 3)
    """
    )
    parser.add_argument('--traindata', type=str, required=True,
                        help='Folder path of the cropped images (train)')
    parser.add_argument('--testdata', type=str, required=True,
                        help='Folder path to the full-scale mammogram images (test)')
    parser.add_argument('-b', '--batch_size', type=int,
                        required=False, default=32, help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, required=False,
                        default=50, help='Number of epochs to run')
    parser.add_argument('--gpu', action='store_true', help='Want GPU ?')
    parser.add_argument('-i', '--interval', type=int, required=False,
                        default=10, help='Iteration interval for display')
    parser.add_argument('-s', '--subset_size', type=int, required=False,
                        default=0, help='Size of the randomly selected subset')
    parser.add_argument('--model_file_name', type=str,
                        required=True, help='Name of model and log files')
    parser.add_argument('--randomize_subset', action='store_true',
                        help='Randomize selected subset in every epoch ?')
    parser.add_argument('-d', '--depth', type=int, required=False,
                        default=3, help='no of conv blocks in the architecture')
    parser.add_argument('-w', '--width', type=int, required=False,
                        default=64, help='no of Filters per conv')
    parser.add_argument('-p', '--phi', type=float,
                        required=False, default=1.5, help='Computation power')
    parser.add_argument('--efficient_net',
                        action='store_true', help='Run Efficient Net')
    parser.add_argument('-iter', '--iterate', type=int, required=False,
                        default=50, help='Grid search iterations for alpha-beta')
    parser.add_argument('-p', '--phase', type=int,
                        required=True, default=1, help='Phase 1 or Phase 2')
    args = parser.parse_args()

    main(args)
