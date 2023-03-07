from torch.utils import data
from dataloader import BUS_Dataset
from solve import solver
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class Args:
    def __init__(self):
        self.name = 'results'
        self.model_name = ''
        self.stage0_data = 'breast_ultrasound_simulation/stage1/DATASET_BUS/stage0'
        self.realUS_data = 'breast_ultrasound_simulation/stage1/DATASET_BUS/images'
        self.batch_size = 64
        self.epochs = 2500
        self.lr = 1e-5
        self.log_step = 5
        self.vis_step = 2
        self.restore = 0
        self.test = 0
        self.beta_reco = 0.9
        self.beta_adv = 0.1
        self.openvino = 0


if __name__ == '__main__':
    args = Args()

    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': 8,  # 8
              'drop_last': True}

    params_test = {'batch_size': 1,
                   'shuffle': False,
                   'num_workers': 1,
                   'drop_last': False}

    STAGE0_DIR = args.stage0_data
    IMAGES_DIR = args.realUS_data

    all_files = os.listdir(STAGE0_DIR)
    Train_IDs = [temp for temp in all_files if temp[-10] != "1"]
    Test_IDs = [temp for temp in all_files if temp[-10] == "1"]

    print(
        "Split to training set and testing set with sizes :",
        len(Train_IDs),
        len(Test_IDs))

    training_set = BUS_Dataset(Train_IDs, STAGE0_DIR, IMAGES_DIR, resize=True)
    testing_set = BUS_Dataset(Test_IDs, STAGE0_DIR, IMAGES_DIR, resize=True)

    training_gen = data.DataLoader(training_set, **params)
    testing_gen = data.DataLoader(testing_set, **params_test)

    solver_ins = solver(args, training_gen, testing_gen, restore=args.restore)
    best_parameters = solver_ins.optimize_bayesian()

    solver_ins.train(num_epochs_to_run=10000, best_param=best_parameters)
