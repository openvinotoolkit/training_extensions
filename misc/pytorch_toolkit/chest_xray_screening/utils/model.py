import torch
from torchvision import models
import torch.nn as nn



class DenseNet121():
    def __init__(self,class_count):

        self.model = models.densenet121(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier=nn.Sequential(nn.Linear(1024, class_count), nn.Sigmoid())

    return self.model
    