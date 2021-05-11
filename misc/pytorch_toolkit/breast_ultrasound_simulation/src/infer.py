import os
import argparse
from torch.utils import data
from dataloader import IVUS_Dataset, BUS_dataset, IVUS3D_Dataset
from solve import solver_inter2d, solver_inter3d


parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True, help='experiment name')
parser.add_argument(
    '--model_name',
    default=None,
    help='model that needs to be loaded')
parser.add_argument(
    '--stage0_data',
    default='../data/stage0',
    help='path to stage 0 output')
parser.add_argument(
    '--realUS_data',
    default='../data/real_images',
    help='path to real ultrasound images')
parser.add_argument(
    '--dilation_factor',
    type=float,
    default=0.0,
    help='controls the resolution only for IVUS2D and IVUS3D')
parser.add_argument(
    '--infer_data',
    default='BUS',
    help='select from IVUS2D, IVUS3D, BUS')
parser.add_argument('--test', type=int, default=1)


args = parser.parse_args()


params_test = {'batch_size': 1,
               'shuffle': False,
               'num_workers': 1,
               'drop_last': False}

STAGE0_DIR = args.stage0_data
IMAGES_DIR = args.realUS_data

if(args.infer_data == 'IVUS2D'):
    print("Preparing dataset for inferecing on 2D IVUS dataset")

    all_files = os.listdir(STAGE0_DIR)
    test_ids = [temp for temp in all_files if temp[6:8] == "09"]
    testing_set = IVUS_Dataset(test_ids, STAGE0_DIR, IMAGES_DIR)

elif(args.infer_data == 'IVUS3D'):
    print("Preparing dataset for inferecing on 3D IVUS dataset")

    all_files = os.listdir(STAGE0_DIR)
    test_ids = [temp for temp in all_files if temp[1:5] == "09_0"]
    testing_set = IVUS3D_Dataset(test_ids, STAGE0_DIR, IMAGES_DIR)


elif(args.infer_data == 'BUS'):
    print("Preparing dataset for inferecing on Breast ultrasound dataset dataset")

    inb_te = os.listdir(STAGE0_DIR)
    file_inb_te = [temp for temp in inb_te]
    testing_set = BUS_dataset(file_inb_te, STAGE0_DIR, resize=True, test=1)


testing_gen = data.DataLoader(testing_set, **params_test)

print("Loaded Dataset")

if(args.infer_data == 'IVUS3D'):
    solver_ins = solver_inter3d(args, test_data=testing_gen, restore=1)
    solver_ins.test()
else:
    if(args.infer_data == 'BUS'):
        args.dilation_factor = 0
    solver_ins = solver_inter2d(args, test_data=testing_gen, restore=1)
    solver_ins.test()
