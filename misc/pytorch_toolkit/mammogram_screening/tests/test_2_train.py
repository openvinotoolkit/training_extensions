import unittest
import os
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
# from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
from mammogram_screening.train_utils.models import UNet, Model2
from mammogram_screening.train_utils.dataloader import Stage1Dataset,Stage2bDataset
from mammogram_screening.train_utils.train_function import train_stage1,train_stage2,train_pos_neg_split
from mammogram_screening.train_utils.downloader import download_data#, prepare_data
from mammogram_screening.train_utils.get_config import get_config
from mammogram_screening.train_utils.transforms import augment_color
from mammogram_screening.stage2.step2_get_predictions_for_all import get_pred_all
from mammogram_screening.stage2.step3_get_patches import get_bags
from mammogram_screening.stage1.data_prep_rbis import data_prep

def create_train_test_for_stage1():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config(action='train', stage='stage1')
            cls.config = config
            data_prep()
            if os.path.exists(config['tr_data_path']):
                tr_data_path = config['tr_data_path']
            else:
                download_data()
                tr_data_path = config['tr_data_path']

            x_train = np.load(tr_data_path, allow_pickle=True)
            x_train = np.repeat(x_train, 4, axis=0)
            train_data = Stage1Dataset(x_train, transform=augment_color)
            cls.train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=10)
            cls.train_bce_list,cls.train_dice_loss_list,cls.train_dice_list =[],[],[]

        def test_trainer(self):
            model = UNet(num_filters=32)
            model.to(self.config['device'])
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=self.config['lr'], momentum=0.9, weight_decay=0.0005)
            for epoch in range(self.config['epochs']):
                train_loss_bce, train_loss_dice, train_dice = train_stage1(model, self.train_loader, optimizer, epoch, self.config['epochs'], self.config['device'], verbose=True)
                self.train_bce_list.append(train_loss_bce)
                self.train_dice_loss_list.append(train_loss_dice)
                self.train_dice_list.append(train_dice)

            self.assertLessEqual(self.train_bce_list[2], self.train_bce_list[0])
            self.assertLessEqual(self.train_dice_loss_list[2], self.train_dice_loss_list[0])
            self.assertGreaterEqual(self.train_dice_list[2], self.train_dice_list[0])
    return TrainerTest

def create_train_test_for_stage2():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config(action='train', stage='stage2')
            cls.config = config
            get_pred_all()
            get_bags()
            if os.path.exists(config['train_bags_path']):
                train_bags_path = config['train_bags_path']
            else:
                download_data()
                train_bags_path = config['train_bags_path']

            x_train = np.load(train_bags_path, allow_pickle=True)
            x_train_pos, x_train_neg = train_pos_neg_split(x_train)
            ratio = len(x_train_neg) // len(x_train_pos)
            if ratio == 0:
                ratio = 1
            x_train_pos = x_train_pos * ratio
            x_train = x_train_pos + x_train_neg
            train_data = Stage2bDataset(x_train, transform=None)
            cls.train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=config['num_workers'])

        def test_trainer(self):
            model = Model2()
            model.to(self.config['device'])
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=self.config['lr'], momentum=0.9, weight_decay=0.0005)
            criterion = nn.BCELoss()
            train_loss_list, train_acc_list, train_auc_list = [], [], []
            for epoch in range(self.config['epochs']):
                train_loss, train_acc, train_auc= train_stage2(model, self.train_loader, criterion, optimizer, epoch, self.config['epochs'], self.config['device'], verbose=True)
                train_loss_list.append(train_loss)
                train_acc_list.append(train_acc)
                train_auc_list.append(train_auc)

            self.assertLessEqual(train_loss_list[4], train_loss_list[0])
            self.assertGreaterEqual(train_acc_list[4], train_acc_list[0])
            self.assertGreaterEqual(train_auc_list[4], train_auc_list[0])
    return TrainerTest


class TestTrainerStage1(create_train_test_for_stage1()):
    'Test case for Stage1'

class TestTrainerStage2(create_train_test_for_stage2()):
    'Test case for Stage2'

if __name__ == '__main__':

    unittest.main()