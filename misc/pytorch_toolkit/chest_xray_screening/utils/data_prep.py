import os
import json
import csv
from pydicom import dcmread
from tqdm import tqdm as tq
import numpy as np
from PIL import Image
import argparse
import pandas as pd

def dcm_to_jpg(args):

    dirpath = args.dpath + '/original'
    savepath = args.dpath +'/processed_data'
    file_list = os.listdir(dirpath)
    for fpath in tq(file_list):
        ds = dcmread(dirpath+fpath)
        np_array = ds.pixel_array
        im = Image.fromarray(np_array)
        name = fpath.split('.dcm')[0]
        im.save(savepath+name+".jpg")

    print(f'Completed')


def mean_sd_calculator(args):
    dirpath = args.dpath + '/original'
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

    file_list = os.listdir(args.dpath+'/processed_data')
    df_class=pd.read_csv(args.dpath+'stage_2_detailed_class_info.csv')
    labels=["Lung Opacity","Normal","No Lung Opacity / Not Normal"]
    image_name = []
    image_label = []
    for file in tq(file_list):
        patient_id = file.split('.jpg')[0]
        tmp=df_class[df_class["patientId"]==patient_id]["class"].values[0]
        idx=labels.index(tmp)
        image_name.append(file)
        image_label.append(idx)

    dict_annotation = {'names':image_name,'labels':image_label}

    with open(args.dpath+'/rsna_annotation.json','w') as f:
        json.dump(dict_annotation,f)

    print(f'Created and saved rsna_annotation.json file.')


def main(args):

    if args.jpg:
        dcm_to_jpg(args)
    if args.msd:
        mean_sd_calculator(args)
    if args.ann:
        create_annotation(args)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jpg",required=False,help="Convert DCM files to jpg",default=False,type=bool)
    parser.add_argument("--msd",required=False,help="Calculate the mean-sd of the dataset",default=False,type=bool)
    parser.add_argument("--ann",required=False,help="Create rsna_annotation.json file",default=False,type=bool)
    parser.add_argument("--dpath",required=True, help="Absolute path to folder containing the dataset",default= None ,type = str)

    args = parser.parse_args()

    main(args)

