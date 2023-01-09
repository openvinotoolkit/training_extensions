from utils import lung_seg
from utils import lung_seg_adv
from utils import patch_classifier
import argparse

def main(args):

    if args.lungseg or args.lungsegadv:
        foldno = args.foldno
        savepath = args.savepath
        jsonpath = args.jsonpath
        datapath = args.datapath
        lungsegpath = args.lungmask
        network = args.network
        if args.epochs:
            if args.lungsegadv:
                lung_seg.train_network(foldno,savepath,jsonpath,datapath,lungsegpath,network,args.epochs,adv=True)
            else:
                lung_seg.train_network(foldno,savepath,jsonpath,datapath,lungsegpath,network,args.epochs)
        else:
            if args.lungsegadv:
                lung_seg.train_network(foldno,savepath,jsonpath,datapath,lungsegpath,network,args.epochs,adv=True)
            else:
                lung_seg.train_network(foldno,savepath,jsonpath,datapath,lungsegpath,network)

    else:
        savepath = args.savepath
        imgpath = args.datapath
        if args.epochs:
            patch_classifier.lungpatch_classifier(savepath,imgpath,args.epochs)
        else:
            patch_classifier.lungpatch_classifier(savepath,imgpath)


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

    main(args)
