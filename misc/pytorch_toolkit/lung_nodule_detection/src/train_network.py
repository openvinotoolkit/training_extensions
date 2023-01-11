from utils import train_stage1
from utils import train_stage2
import argparse

def main(config):

    if config["lungseg"] or config["lungsegadv"]:
        train_stage1.train_network(config)
    else:
        train_stage2.lungpatch_classifier(config)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Select action to be performed')

    parser.add_argument('--lungseg', default=False, action='store_true',
                        help='To Train lung segmentation')
    parser.add_argument('--lungsegadv', default=False, action='store_true',
                        help='To train lung seg network adversarially')
    parser.add_argument('--patchclass', default=False, action='store_true',
                        help='Train network to classify patch')

    parser.add_argument('--savepath',
                        help='Folder location to save the files')
    parser.add_argument('--foldno',
                        help='Fold number')
    parser.add_argument('--jsonpath',
                        help='Folder location where jsons are stored')
    parser.add_argument('--datapath',
                        help='Folder location where img and masks are stored')
    parser.add_argument('--lungmask',
                        help='Folder location where lung masks are stored')
    parser.add_argument('--network',
                        help='Network to be trained')
    parser.add_argument('--epochs',
                        help='Number of epochs')


    args= parser.parse_args()

    configs = {
        "lungseg": args.lungseg,
        "lungsegadv": args.lungsegadv,
        "patchclass": args.patchclass,
        "savepath": args.savepath,
        "foldno": args.foldno,
        "jsonpath": args.jsonpath,
        "datapath": args.datapath,
        "lungmask": args.lungmask,
        "network": args.network,
        "epochs": args.epochs
    }

    main(configs)
