import argparse
import os
from torch.utils import data
from dataloader import IVUS_Dataset
from solve import solver


parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True, help='experiment name')
parser.add_argument('--model_name', help='model that needs to be loaded')
parser.add_argument(
    '--stage0_data',
    default='../data/stage0',
    help='path to stage 0 output')
parser.add_argument(
    '--realUS_data',
    default='../data/real_images',
    help='path to real ultrasound images')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument(
    '--log_step',
    type=int,
    default=5,
    help="Logging frequency, set -1 for no logging")
parser.add_argument('--vis_step', type=int, default=2,
                    help="Tensorboard visualization frequency")
parser.add_argument(
    '--restore',
    type=int,
    default=0,
    help="restore a model for continuing training")
parser.add_argument('--test', type=int, default=0)
parser.add_argument('--beta_reco', type=float, default=1)
parser.add_argument('--beta_adv', type=float, default=0.02)


args = parser.parse_args()


params = {'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 8,
          'drop_last': True}

params_test = {'batch_size': 1,
               'shuffle': False,
               'num_workers': 1,
               'drop_last': False}

STAGE0_DIR = args.stage0_data
IMAGES_DIR = args.realUS_data


all_files = os.listdir(STAGE0_DIR)
Train_IDs = [temp for temp in all_files if temp[6:8] != "09"]
Test_IDs = [temp for temp in all_files if temp[6:8] == "09"]


print(
    "Split to training set and testing set with sizes :",
    len(Train_IDs),
    len(Test_IDs))

training_set = IVUS_Dataset(Train_IDs, STAGE0_DIR, IMAGES_DIR, resize=True)
testing_set = IVUS_Dataset(Test_IDs, STAGE0_DIR, IMAGES_DIR, resize=True)


training_gen = data.DataLoader(training_set, **params)
testing_gen = data.DataLoader(testing_set, **params_test)


solver_ins = solver(args, training_gen, testing_gen, restore=args.restore)
solver_ins.train()
