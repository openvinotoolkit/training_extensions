import argparse
from .utils import visualize
from .utils import generate_slices
from .utils import create_folds
from .utils import generate_patches


def main(args):

    if args.genslice:
        dataset_path = args.datasetpath
        save_path = args.savepath
        masktype = args.masktype
        if masktype == 'nodule':
            generate_slices.extract_slices(dataset_path,save_path,masktype)
        else:
            generate_slices.generate_lungseg(dataset_path,save_path)

    elif args.createfolds:
        data_path = args.datapath
        save_path = args.savepath
        dataset_path = args.datasetpath

        create_folds.positive_negative_classifier(data_path,save_path)
        dict_subset = create_folds.subset_classifier(dataset_path,save_path)
        create_folds.assign_folds(dict_subset,save_path)
        if args.additional:
            create_folds.create_balanced_dataset(save_path,data_path,additional=True)
        else:
            create_folds.create_balanced_dataset(save_path,data_path)


    elif args.genpatch:
        jsonpath = args.jsonpath
        foldno = args.foldno
        category = args.category
        data_path = args.datapath
        lungsegpath = args.lungsegpath
        savepath = args.savepath
        patchtype = args.patchtype
        if patchtype == 'positive':
            generate_patches.generate_patchlist(jsonpath,patchtype,foldno)
            generate_patches.generate_positive_patch(jsonpath,foldno,data_path,savepath,category)
        else:
            generate_patches.generate_patchlist(jsonpath,patchtype,foldno)
            generate_patches.generate_negative_patch(jsonpath,foldno,data_path,lungsegpath,savepath,category)


    elif args.visualize:
        seriesuid = args.seriesuid
        slice_num = args.sliceno
        data_path = args.datapath
        savepath = args.savepath

        visualize.visualize_data(seriesuid,slice_num,data_path,savepath)

    else:
        print('Arguments not passed. Use -h for help')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Select action to be performed')

    parser.add_argument('--genslice', default=False, action='store_true',
                        help='To create slices from 3D volume')
    parser.add_argument('--createfolds', default=False, action='store_true',
                        help='Split dataset into 10 folds')
    parser.add_argument('--genpatch', default=False, action='store_true',
                        help='To create patches from 2D slices')
    parser.add_argument('--visualize', default=False, action='store_true',
                        help='Visualize any one of the slices')
    parser.add_argument('--savepath',
                        help='Folder location to save the files')
    parser.add_argument('--masktype',
                        help='Type of mask to be generated. ie, nodule or lung')
    parser.add_argument('--datasetpath',
                        help='Folder location of downloaded dataset')
    parser.add_argument('--foldno',
                        help='Fold number')
    parser.add_argument('--additional', default=False, action='store_true',
                        help='Add additional slices')
    parser.add_argument('--category',
                        help='Category of data.[trainset,valset,testset]')
    parser.add_argument('--jsonpath',
                        help='Folder location where jsons are stored')
    parser.add_argument('--datapath',
                        help='Folder location containing img and mask folders')
    parser.add_argument('--lungsegpath',
                        help='Folder containing lung segmentation mask')
    parser.add_argument('--patchtype',
                        help='positive or negative')
    parser.add_argument('--sliceno',
                        help='Slice number to visualize')
    parser.add_argument('--seriesuid',
                        help='Seriesuid of slice to visualize')

    args=parser.parse_args()

    main(args)
