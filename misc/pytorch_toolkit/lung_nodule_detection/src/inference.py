from .utils import infer_stage1
from .utils import infer_stage2
import argparse

def main(config):

    if config["lungseg"]:
        foldno = config["foldno"]
        savepath = config["savepath"]
        jsonpath = config["jsonpath"]
        network = config["network"]
        infer_stage1.infer_lungseg(foldno,savepath,network,jsonpath)
    else:
        savepath = config["savepath"]
        imgpath = config["imgpath"]
        infer_stage2.lungpatch_classifier(savepath,imgpath)

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

    args = parser.parse_args()

    configs = {
        "lungseg": args.lungseg,
        "patchclass": args.patchclass,
        "savepath" : args.savepath,
        "foldno" : args.foldno,
        "jsonpath" : args.jsonpath,
        "imgpath" : args.imgpath,
        "network" : args.network
    }

    main(configs)
