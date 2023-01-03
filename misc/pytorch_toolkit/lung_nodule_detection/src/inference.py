from .utils import infer_lung_seg
from .utils import infer_patch_classifier
import argparse

def main(args):

    if args.lungseg:
        foldno = args.foldno
        savepath = args.savepath
        jsonpath = args.jsonpath
        network = args.network
        infer_lung_seg.infer_lungseg(foldno,savepath,network,jsonpath)
    else:
        savepath = args.savepath
        imgpath = args.imgpath
        infer_patch_classifier.lungpatch_classifier(savepath,imgpath)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Select action to be performed')

    parser.add_argument('--lungseg', default=False, action='store_true',
                        help='To test lung segmentation')
    parser.add_argument('--patchclass', default=False, action='store_true',
                        help='test network to classify patch')
    parser.add_argument('--savepath',
                        help='Folder location to save the files')
    parser.add_argument('--foldno',
                        help='Fold number')
    parser.add_argument('--jsonpath',
                        help='Folder location where jsons are stored')
    parser.add_argument('--imgpath',
                        help='Folder location where test images are stored')
    parser.add_argument('--network',
                        help='Network to be trained')						

    args= parser.parse_args()

    main(args)
