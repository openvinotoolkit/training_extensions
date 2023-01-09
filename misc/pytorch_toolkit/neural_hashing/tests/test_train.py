import unittest
import os
import torch
from torchvision import transforms
from src.train import Trainer
from src.utils.dataloader import DataloderImg
from src.utils.network import Encoder, Classifier1, Discriminator
from src.utils.get_config import get_config
from src.utils.downloader import download_checkpoint, download_data

def create_train_test_for_encoder():
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config(action = 'train')
            cls.config = config
            current_dir =  os.path.abspath(os.path.dirname(__file__))
            parent_dir = os.path.abspath(current_dir + "/../")
            train_path = parent_dir + config['train_path']
            val_path =  parent_dir + config["val_path"]
            print(train_path)
            if not os.path.exists(train_path):
                download_data()
            batch_size = config["batch_size"]
            transform = transforms.Compose([
            transforms.Resize((28, 28), transforms.InterpolationMode.BICUBIC),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
             ])
            #Data loading
            trainset = DataloderImg(train_path, transform=transform, target_transform=None)
            cls.trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=batch_size, num_workers=4)
            valset = DataloderImg(val_path, transform=transform, target_transform=None)
            cls.valloader = torch.utils.data.DataLoader(valset, shuffle=True,
                                                    batch_size=int(batch_size/2), num_workers=4)
        def test_config(self):
            self.assertGreaterEqual(self.config["lr"], 1e-8)
            self.assertEqual(self.config["class_count"], 4)
            self.assertEqual(self.config["zSize"], 48)

        def test_trainer(self):
            self.device = self.config["device"]
            self.encoder = Encoder(self.config["zSize"]).to(self.device)
            self.classifier = Classifier1(self.config["class_count"]).to(self.device)
            self.discriminator = Discriminator(self.config["zSize"]).to(self.device)
            current_dir =  os.path.abspath(os.path.dirname(__file__))
            parent_dir = os.path.abspath(current_dir + "/../")
            model_path = os.path.join(parent_dir + self.config['savepath'], 'encoder-100.pkl')
            self.encoder.load_state_dict(torch.load(model_path,map_location=self.device), strict=False)
            if not os.path.exists(model_path):
                download_checkpoint()
            self.similarity_loss_dict, self.relational_loss_dict = Trainer(
                self.encoder, self.classifier, self.discriminator,
                self.trainloader, self.valloader, self.config["lr"], self.config["max_epoch"],
                self.config["batch_size"], self.config["zSize"], self.config['alpha1'],
                self.config['alpha2'], self.config['beta'], self.config['gamma']
                )
            self.assertLessEqual(self.similarity_loss_dict[13], self.similarity_loss_dict[1])
            self.assertLessEqual(self.relational_loss_dict[13], self.relational_loss_dict[1])


    return TrainerTest


class TestTrainer(create_train_test_for_encoder()):
    'Test case for model'


if __name__ == '__main__':

    unittest.main()
