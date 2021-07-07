from torchvision import models
from torch import nn
from utils.generate import give_model

class DenseNet121():
    def __init__(self,class_count):
        self.model = models.densenet121(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier=nn.Sequential(nn.Linear(1024, class_count), nn.Sigmoid())
    def __new__(cls):
        return self.model

class DenseNet121Eff():
    def __init__(self,alpha, beta, class_count):
        self.model = give_model(alpha, beta, class_count)
        self.model = nn.Sequential(self.model, nn.Sigmoid())
    def __new__(cls):
        return self.model
