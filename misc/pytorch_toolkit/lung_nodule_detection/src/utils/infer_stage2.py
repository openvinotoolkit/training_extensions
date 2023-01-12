import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils import data
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm as tq
import json
from sklearn.metrics import confusion_matrix
from .data_loader import LungPatchDataLoader
from .models import LeNet
from .utils import load_inference_model

def lungpatch_classifier(config, run_type):
    imgpath = config["imgpath"]
    modelpath = config["modelpath"]
    jsonpath = config["jsonpath"]

    with open(jsonpath) as f:
        json_file = json.load(f)

    testDset = LungPatchDataLoader(imgpath,json_file,is_transform=True,split="test_set")
    testDataLoader = data.DataLoader(testDset,batch_size=1,shuffle=True,num_workers=4,pin_memory=True)
    classification_model_loadPath = modelpath

    use_gpu = torch.cuda.is_available()
    if run_type == 'pytorch':
        net = LeNet()
        if use_gpu:
            net = net.cuda()
        net.load_state_dict(torch.load(classification_model_loadPath+'lenet_best.pt'))
    elif run_type == 'onnx':
        net = load_inference_model(config, run_type='onnx')
    else:
        net = load_inference_model(config, run_type='ir')


    optimizer = optim.Adam(net.parameters(), lr = 1e-4, weight_decay = 1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    testRunningCorrects = 0
    testRunningLoss = 0
    testBatches = 0
    pred_arr = []
    label_arr = []
    for data1 in tq(testDataLoader):
        inputs, label = data1

        if run_type == 'pytorch':
            if use_gpu:
                inputs = inputs.cuda()
                label = label.float()
                label = label.cuda()
            net_out = net(Variable(inputs))
        elif run_type == 'ir':
            net_out = net.infer(inputs={'input': inputs})['output']
            net_out = torch.tensor(net_out)
        else:
            ort_inputs = {net.get_inputs()[0].name: to_numpy(inputs)}
            net_out = net.run(None, ort_inputs)
            net_out = np.array(net_out)
            net_out = torch.tensor(net_out)

        net_loss = criterion(net_out,label)
        preds = torch.zeros(net_out.shape).cuda()
        preds[net_out > 0.5] = 1
        preds[net_out <= 0.5] = 0

        testRunningLoss += net_loss.item()
        testRunningCorrects += torch.sum(preds == label.data.float())

        for i,j in zip(preds.cpu().numpy(),label.cpu().numpy()):
            pred_arr.append(i)
            label_arr.append(j)

        testBatches += 1
        # if testBatches>0:
        #     break

    testepoch_loss = testRunningLoss/testBatches
    testepoch_acc = 100*(int(testRunningCorrects)/len(pred_arr))

    print(' Loss: {:.4f} | accuracy: {:.4f} '.format(
             testepoch_loss,testepoch_acc))

    return testepoch_acc
