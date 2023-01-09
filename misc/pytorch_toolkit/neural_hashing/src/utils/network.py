import torch
from torch import nn
import torch.nn.functional as F

def load_checkpoint(model, checkpoint):
    if checkpoint is not None:
        model_checkpoint = torch.load(checkpoint)
        model.load_state_dict(model_checkpoint)



class Encoder(nn.Module):
    def __init__(self, zsize):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, zsize)
        '''self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()'''

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)

        x_c = F.relu(self.fc1(y))#x
        x_out = torch.tanh(self.fc2(x_c))#h
        #print(x_c)
        #print(x_out)
        return x_out, x_c  # ,indices1,indices2,indices3


class Classifier1(nn.Module):
    def __init__(self, numclasses):
        super().__init__()
        # self.conv1d = nn.Conv1d(2,1,1)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 64)
        self.fc3 = nn.Linear(64, numclasses)

    def forward(self, x):
        # x = F.relu(self.conv1d(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=0)
        return x


class Discriminator(nn.Module):
    def __init__(self, zsize):
        super().__init__()
        self.conv1d = nn.Conv1d(2, 1, 1)
        self.fc1 = nn.Linear(zsize, 32)
        self.fc3 = nn.Linear(32, 1)
        #self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1d(x))
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc3(x))
        x = self.fc3(x)
        return x
