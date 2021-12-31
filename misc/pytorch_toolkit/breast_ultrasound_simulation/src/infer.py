import os
from torch.utils import data
from dataloader import Kaggle_BUS_Dataset
from solve import solver_inter2d
from utils.read_config import load_config


net_config_path = "configs/net_config.json"
net_config = load_config(net_config_path)
data_config = net_config['data']
gen_config = net_config['general']
infer_config = net_config['infer']

params_test = {'batch_size': 1,
               'shuffle': False,
               'num_workers': 1,
               'drop_last': False}

STAGE0_DIR = data_config["stage0_data"]
IMAGES_DIR = data_config["realUS_data"]

if infer_config["infer_data"] == 'BUS':
    print("Preparing dataset for inferecing on 2D BUS dataset")

    all_files = os.listdir(STAGE0_DIR)
    test_ids = [temp for temp in all_files if temp[6:8] == "09"]
    testing_set = Kaggle_BUS_Dataset(test_ids, STAGE0_DIR, IMAGES_DIR)

testing_gen = data.DataLoader(testing_set, **params_test)


print("Loaded Dataset")

if infer_config["infer_data"] == 'BUS':
    solver_ins = solver_inter2d(infer_config, test_data=testing_gen, test_flag=gen_config['test_flag'], restore=1)
    solver_ins.test()
