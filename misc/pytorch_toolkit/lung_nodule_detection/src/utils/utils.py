import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import json
from .models import LeNet, R2U_Net, SUMNet, U_Net


def ch_shuffle(x):
    shuffIdx1 = torch.from_numpy(np.random.randint(0,2,x.size(0)))
    shuffIdx2 = 1-shuffIdx1
    d_in = torch.Tensor(x.size()).cuda()
    d_in[:,shuffIdx1] = x[:,0]
    d_in[:,shuffIdx2] = x[:,1]
    shuffLabel = torch.cat((shuffIdx1.unsqueeze(1),shuffIdx2.unsqueeze(1)),dim=1)
    return d_in, shuffLabel

def dice_coefficient(pred1, target):
    smooth = 1e-15
    pred = torch.argmax(pred1,dim=1)
    num = pred.size()[0]
    pred_1_hot = torch.eye(3)[pred.squeeze(1)].cuda()
    pred_1_hot = pred_1_hot.permute(0, 3, 1, 2).float()

    target_1_hot = torch.eye(3)[target].cuda()
    target_1_hot = target_1_hot.permute(0,3, 1, 2).float()

    m1_1 = pred_1_hot[:,1,:,:].view(num, -1).float()
    m2_1 = target_1_hot[:,1,:,:].view(num, -1).float()
    m1_2 = pred_1_hot[:,2,:,:].view(num, -1).float()
    m2_2 = target_1_hot[:,2,:,:].view(num, -1).float()

    intersection_1 = (m1_1*m2_1).sum(1)
    intersection_2 = (m1_2*m2_2).sum(1)
    union_1 = (m1_1+m2_1).sum(1) + smooth - intersection_1
    union_2 = (m1_2+m2_2).sum(1) + smooth - intersection_2
    score_1 = intersection_1/union_1

    return [score_1.mean()]

def load_model(network):

    if network == 'unet':
        net = U_Net(img_ch=1,output_ch=2)
    elif network == 'r2unet':
        net = R2U_Net(img_ch=1,output_ch=2)
    elif network == 'sumnet':
        net = SUMNet(in_ch=1,out_ch=2)
    else:
        net = LeNet()
    return net

def load_checkpoint(model, checkpoint):
    if checkpoint is not None:
        model_checkpoint = torch.load(checkpoint)
        model.load_state_dict(model_checkpoint)
    else:
        model.state_dict()

def plot_graphs(
    train_values, valid_values,
    save_path, x_label, y_label,
    plot_title, save_name):

    plt.figure()
    plt.plot(range(len(train_values)),train_values,'-r',label='Train')
    plt.plot(range(len(valid_values)),valid_values,'-g',label='Valid')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title)
    plt.legend()
    plt.savefig(os.path.join(save_path, save_name))
    plt.close()

def create_dummy_json_file(json_path,stage):
    test_data_path = os.path.split(json_path)[0]
    if stage == 1:
        img_path = os.path.join(test_data_path,'stage1','img')
    else:
        img_path = os.path.join(test_data_path,'stage2','img')
    file_list = os.listdir(img_path)
    train_list = file_list[:7]
    valid_list = file_list[7:10]
    test_list = file_list[10:15]
    dummy_dict = {
        "train_set":train_list,
        "valid_set": valid_list,
        "test_set": test_list
    }

    with open(json_path, 'w') as h:
        json.dump(dummy_dict, h)
