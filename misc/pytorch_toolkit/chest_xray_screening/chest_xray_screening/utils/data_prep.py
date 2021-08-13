import os
import json
from pydicom import dcmread
from tqdm import tqdm as tq
import numpy as np
from PIL import Image
import argparse
import pandas as pd

def dcm_to_jpg(args):

    dirpath = os.path.join(args.dpath, 'original/')
    savepath = os.path.join(args.dpath, 'processed_data/')
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    file_list = os.listdir(dirpath)
    for fpath in tq(file_list):
        ds = dcmread(os.path.join(dirpath, fpath))
        np_array = ds.pixel_array
        im = Image.fromarray(np_array)
        name = fpath.split('.dcm')[0]
        im.save(os.path.join(savepath, name + ".jpg"))

    print('Completed')


def mean_sd_calculator(args):
    dirpath = os.path.join(args.dpath, 'original/')
    file_list = os.listdir(dirpath)
    mean_list = []
    sd_list = []
    for fpath in tq(file_list):
        ds = dcmread(dirpath+fpath)
        np_array = ds.pixel_array
        mean = np.mean(np_array)
        mean_list.append(mean)
        sd = np.std(np_array)
        sd_list.append(sd)

    print(f'Mean: {np.mean(mean_list)}')
    print(f'Standard Deviation: {np.std(sd_list)}')

def create_annotation(args):

    file_list = os.listdir(os.path.join(args.dpath, 'processed_data/'))
    df_class=pd.read_csv(os.path.join(args.dpath, 'stage_2_detailed_class_info.csv'))
    labels=["Lung Opacity","Normal","No Lung Opacity / Not Normal"]
    dict_annotation = {}
    for file in tq(file_list):
        patient_id = file.split('.jpg')[0]
        tmp = df_class[df_class["patientId"] == patient_id]["class"].values[0]
        idx = labels.index(tmp)
        dict_annotation[str(file)] = idx


    with open(os.path.join(args.dpath, 'rsna_annotation.json'), 'w') as f:
        json.dump(dict_annotation, f)

    print('Created and saved rsna_annotation.json file.')


def main(args):

    if args.jpg:
        dcm_to_jpg(args)
    if args.msd:
        mean_sd_calculator(args)
    if args.ann:
        create_annotation(args)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jpg",
        required=False,
        help="Convert DCM files to jpg",
        default=False,
        action='store_true')
    parser.add_argument(
        "--msd",
        required=False,
        help="Calculate the mean-sd of the dataset",
        default=False,
        action='store_true')
    parser.add_argument(
        "--ann",
        required=False,
        help="Create rsna_annotation.json file",
        default=False,
        action='store_true')
    parser.add_argument(
        "--dpath",
        required=True,
        help="Absolute path to folder containing the dataset",
        default= None,
        type = str)

    custom_args = parser.parse_args()

    main(custom_args)
