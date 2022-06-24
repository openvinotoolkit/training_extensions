import unittest
import os
import numpy as np
from torch.utils.data import DataLoader
from src.utils.get_config import get_config
from src.utils.data_loader import CustomDatasetPhase1, CustomDatasetPhase2

def augment_color(self, image):
    # modifying colour tones on input image :-
    channel_ranges = (self.red_range, self.green_range, self.blue_range)
    for channel, channel_range in enumerate(channel_ranges):
        if not channel_range:
            continue  # no range set, so don't change that channel
        scale = np.random.uniform(channel_range[0], channel_range[1])
        image[:, :, channel] = image[:, :, channel] * scale
    image = np.clip(image, 0, 255)
    return image

def create_train_test_for_phase1():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config(action='train', phase=1)
            cls.config = config
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
            optimizer = optim.SGD(filter(
                                        lambda p: p.requires_grad,
                                        model.parameters()),
                                        lr=self.config['lr'],
                                        momentum=0.9,
                                        weight_decay=0.0005)
            for epoch in range(self.config['epochs']):
                train_loss_bce, train_loss_dice, train_dice = train_stage1(model,
                                                                        self.train_loader,
                                                                        optimizer, epoch,
                                                                        self.config['epochs'],
                                                                        self.config['device'],
                                                                        verbose=True)
                self.train_bce_list.append(train_loss_bce)
                self.train_dice_loss_list.append(train_loss_dice)
                self.train_dice_list.append(train_dice)

            self.assertLessEqual(self.train_bce_list[2], self.train_bce_list[0])
            self.assertLessEqual(self.train_dice_loss_list[2], self.train_dice_loss_list[0])
            self.assertGreaterEqual(self.train_dice_list[2], self.train_dice_list[0])
    return TrainerTest