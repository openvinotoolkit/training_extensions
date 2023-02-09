from utils.train_utils import train_model
import argparse
def main(args):
    config = {
        'data': args.data,
        'split': args.split,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'gpu': args.gpu,
        'model_file_name': args.model_file_name,
        'gnn': args.gnn,
    }
    train_model(config)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        """
    Training script.
    Provide the folder paths which contains dataset and split.npz file.
    Provide a filename prefix to help differentiate model/log from different runs with different configurations.
    (Optionally) You may choose to provide batch size (--batch_size), number of epochs (--epoch),
    (Optionally) You may choose to use GPU (--gpu).
    (Optionally) You may choose to use GNN (--gnn).
    (optionally) You may choose to use resnet/densenet/xception for backbone. Default is resnet.
    """
    )
    parser.add_argument('--data', type=str, required=True,
                        help='Folder path of dataset')
    parser.add_argument('--split', type=str, required=True,
                        help='Folder path to the full-scale mammogram images (test)')
    parser.add_argument('-b', '--batch_size', type=int,
                        required=False, default=32, help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, required=False,
                        default=15, help='Max number of epochs to run')
    parser.add_argument('--gpu', action='store_true', help='Want GPU ?')
    parser.add_argument('--model_file_name', type=str,
                        required=True, help='Name of model and log files')
    parser.add_argument('--gnn', action='store_true', help='using gnn or not?')
    parser.add_argument('--backbone', type=str, required=False, default='resnet', help='backbone model')
    arguments = parser.parse_args()

    main(arguments)
