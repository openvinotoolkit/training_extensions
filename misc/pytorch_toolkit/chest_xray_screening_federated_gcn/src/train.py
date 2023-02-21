from utils.train_utils_cnn import train_model
from utils.train_utils_gnn import train_model as train_model_gnn
import argparse
def main(args):
    config = {
        'data': args.data,
        'split_npz': args.split_npz,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'gpu': args.gpu,
        'gnn': args.gnn,
        'backbone': args.backbone,
        'lr': args.lr,
        'checkpoint': args.checkpoint,
        'savepath': args.savepath,
    }
    if args.gnn == 'True':
        train_model_gnn(config)
    else:
        train_model(config)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        """
    Training script.
    Provide the folder paths which contains dataset.
    Provide the folder paths which contains split.npz file.
    (Optionally) You may choose to provide batch size (--batch_size), number of epochs (--epoch),
    (Optionally) You may choose to use GPU (--gpu).
    (Optionally) You may choose to use GNN (--gnn).
    (Optionally) You may choose to use resnet/densenet/xception for backbone. Default is resnet.(--backbone)
    (Optionally) You may choose to provide checkpoint, path for model weights to resume training from(--checkpoint).
    (Optionally) You may choose to provide path to save model weights(--savepath).
    (Optionally) You may choose to provide learning rate(--lr).
    """
    )
    parser.add_argument('--data', type=str, required=True,
                        help='Folder path of dataset')
    parser.add_argument('--split_npz', type=str, required=True,
                        help='path to the split.npz file')
    parser.add_argument('-b', '--batch_size', type=int,
                        required=False, default=15, help='Batch size used for training and validation')
    parser.add_argument('-e', '--epochs', type=int, required=False,
                        default=15, help='Max number of epochs to run')
    parser.add_argument('--gpu', type=str, help='Want GPU ?', required=False, default='False')
    parser.add_argument('--checkpoint', type=str,
                        required=False, help='Path of model weights to load for resuming training')
    parser.add_argument('--gnn', type=str, help='using gnn or not?',
                        required=False, default='False')
    parser.add_argument('--backbone', type=str, required=False, default='resnet',
                        help='backbone model to be used from resnet/densenet/xception, by default resnet is used')
    parser.add_argument('--lr', type=float, required=False, default=0.0001, help='learning rate')
    parser.add_argument('--savepath', type=str, required=False, default='model.pth', help='path to save the model')
    arguments = parser.parse_args()

    main(arguments)
