from torchvision import models
import torch
from torch import nn
from .generate import give_model

def load_checkpoint(model, checkpoint):
    if checkpoint is not None:
        model_checkpoint = torch.load(checkpoint)
        model.load_state_dict(model_checkpoint['state_dict'])
    else:
        model.state_dict()

class DenseNet121(nn.Module):
    def __init__(self,class_count):
        super().__init__()
        self.model = models.densenet121(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier=nn.Sequential(nn.Linear(1024, class_count), nn.Sigmoid())
    def forward(self, x):
        x = self.model(x)
        return x



class DenseNet121Eff(nn.Module):
    def __init__(self, alpha, beta, class_count):
        super().__init__()
        self.model = give_model(alpha, beta, class_count)
        self.model = nn.Sequential(self.model, nn.Sigmoid())
    def forward(self, x):
        x = self.model(x)
        return x
