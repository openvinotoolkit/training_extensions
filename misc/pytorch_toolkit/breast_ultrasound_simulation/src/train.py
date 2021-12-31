import os
from torch.utils import data
from dataloader import Kaggle_BUS_Dataset
from solve import solver
from utils.read_config import load_config


if __name__ == '__main__':

    net_config_path = "../../configs/net_config.json"
    net_config = load_config(net_config_path)
    train_config = net_config['train']
    test_config = net_config['test']
    gen_config = net_config['general']


    params = {'batch_size': train_config['batch_size'],
              'shuffle': True,
              'num_workers': train_config['num_workers'],
              'drop_last': True}

    params_test = {'batch_size': test_config['batch_size'],
                   'shuffle': False,
                   'num_workers': test_config['num_workers'],
                   'drop_last': False}

    STAGE0_DIR = gen_config['stage0_data']
    IMAGES_DIR = gen_config['realUS_data']

    all_files = os.listdir(STAGE0_DIR)
    train_IDs = [temp for temp in all_files if temp[-11:-10] != "1"]
    test_IDs = [temp for temp in all_files if temp[-11:-10] == "1"]

    print(
        "Split to training set and testing set with sizes :",
        len(train_IDs),
        len(test_IDs))

    training_set = Kaggle_BUS_Dataset(train_IDs, STAGE0_DIR, IMAGES_DIR, resize=True)
    testing_set = Kaggle_BUS_Dataset(test_IDs, STAGE0_DIR, IMAGES_DIR, resize=True)

    training_gen = data.DataLoader(training_set, **params)
    testing_gen = data.DataLoader(testing_set, **params_test)

    solver_ins = solver(train_config, gen_config, training_gen, testing_gen, restore=train_config['restore'])
    best_parameters = solver_ins.optimize_bayesian()

    solver_ins.train(num_epochs_to_run=train_config['epochs'], best_param=best_parameters)
