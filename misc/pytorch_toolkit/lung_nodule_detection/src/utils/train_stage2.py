import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils import data
from torch.autograd import Variable
import time
import os
from tqdm import tqdm as tq
import json
from .data_loader import LungPatchDataLoader
from .models import LeNet
from .utils import plot_graphs


def lungpatch_classifier(config):
    """Trains network to classify patches based on the presence of nodule

    Parameters
    ----------
    save_path: str
        Folder location to save the plots and model
    img_path:
        Folder location where patch images are stored.
    lrate: int,Default = 1e-4
        Learning rate
    epochs: int, default = 35
        Total epochs

    Returns
    -------

    None
    """
    save_path = config["savepath"]
    img_path = config["imgpath"]
    lrate  = config["lrate"]
    epochs = config["epochs"]
    json_path = config["jsonpath"]

    with open(json_path) as f:
        json_file = json.load(f)

    trainDset = LungPatchDataLoader(imgpath=img_path,json_file=json_file,is_transform=True,split="train_set")
    valDset = LungPatchDataLoader(imgpath=img_path,json_file=json_file,is_transform=True,split="valid_set")
    trainDataLoader = data.DataLoader(trainDset,batch_size=16,shuffle=True,num_workers=4,pin_memory=True)
    validDataLoader = data.DataLoader(valDset,batch_size=16,shuffle=True,num_workers=4,pin_memory=True)

    save_path = save_path
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    net = LeNet()

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        net = net.cuda()

    optimizer = optim.Adam(net.parameters(), lr = lrate, weight_decay = 1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    epochs = epochs
    trainLoss = []
    validLoss = []
    trainAcc = []
    validAcc = []
    start = time.time()
    bestValidAcc = 0.0

    for epoch in range(epochs):
        epochStart = time.time()
        trainRunningLoss = 0
        validRunningLoss = 0
        trainRunningCorrects = 0
        validRunningCorrects = 0
        trainBatches = 0
        validBatches = 0

        net.train(True)

        for data1 in tq(trainDataLoader):
            img, label = data1
            if use_gpu:
                inputs = img.cuda()
                label = label.cuda()

            net_out = net(Variable(inputs))

            net_loss = criterion(net_out,label.float())
            preds = torch.zeros(net_out.shape).cuda()
            preds[net_out > 0.5] = 1
            preds[net_out <= 0.5] = 0

            optimizer.zero_grad()

            net_loss.backward()

            optimizer.step()

            trainRunningLoss += net_loss.item()
            for i in range(len(preds[:,0])):
                if preds[:,0][i] == label[:,0][i].float():
                    trainRunningCorrects += 1

            trainBatches += 1
    #         if trainBatches>1:
    #             break

        trainepoch_loss = trainRunningLoss/trainBatches
        trainepoch_acc = 100*(int(trainRunningCorrects)/32594)
        trainLoss.append(trainepoch_loss)
        trainAcc.append(trainepoch_acc)

        print(f'Epoch: {epoch+1}/{epochs}, Train Loss:{trainepoch_loss}, Train acc:{trainepoch_acc}')

        with torch.no_grad():
            for data1 in tq(validDataLoader):
                img, label = data1
                if use_gpu:
                    inputs = img.cuda()
                    label = label.float()
                    label = label.cuda()

                net_out = net(Variable(inputs))

                net_loss = criterion(net_out,label)
                preds = torch.zeros(net_out.shape).cuda()
                preds[net_out > 0.5] = 1
                preds[net_out <= 0.5] = 0

                validRunningLoss += net_loss.item()
                for i in range(len(preds[:,0])):
                    if preds[:,0][i] == label[:,0][i].float():
                        validRunningCorrects += 1

                validBatches += 1
    #             if validBatches>10:
    #                 break

            validepoch_loss = validRunningLoss/validBatches
            validepoch_acc = 100*(int(validRunningCorrects)/3666)
            validLoss.append(validepoch_loss)
            validAcc.append(validepoch_acc)

            print(f'Epoch: {epoch} Loss: {validepoch_loss} | accuracy: {validepoch_acc}')

        if validepoch_acc > bestValidAcc:
            bestValidAcc = validepoch_acc
            torch.save(net.state_dict(), save_path+'lenet_best.pt')

        scheduler.step(validepoch_loss)

        plot_graphs(
        train_values=trainLoss, valid_values=validLoss,
        save_path=save_path, x_label='Epochs', y_label='Loss',
        plot_title='Loss plot', save_name='LossPlot.png')

        epochEnd = time.time()-epochStart
        print(f'Epoch: {epoch+1}/{epochs} | Train Loss: {trainepoch_loss} | Valid Loss: {validepoch_loss}')
        print('Accuracy | Train_acc {trainepoch_acc} | Valid_acc  {validepoch_acc} |')

        print(f'Time: {epochEnd//60}m {epochEnd%60}s')


        print(f'Saving losses')

        torch.save(trainLoss, save_path+'trainLoss.pt')
        torch.save(validLoss, save_path+'validLoss.pt')
        torch.save(trainAcc, save_path+'train_acc.pt')
        torch.save(validAcc, save_path+'valid_acc.pt')

    #     if epoch>1:
    #         break

    end = time.time()-start
    print(f'Training completed in: {end//60}m {end%60}s')
    plot_graphs(
    train_values=trainLoss, valid_values=validLoss,
    save_path=save_path, x_label='Epochs', y_label='Loss',
    plot_title='Loss plot', save_name='trainLossFinal.png')

    plot_graphs(
    train_values=trainAcc, valid_values=validAcc,
    save_path=save_path, x_label='Epochs', y_label='Accuracy',
    plot_title='Accuracy Plot', save_name='acc_plot.png')

    return trainLoss
