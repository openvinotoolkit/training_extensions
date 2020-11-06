import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import mobilenetv3_large

model = mobilenetv3_large(prob_dropout=0.5, type_dropout='bernoulli')
i = 0
print(len(list(model.parameters())))
