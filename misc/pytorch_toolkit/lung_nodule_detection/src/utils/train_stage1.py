import torch
import torch.nn as nn
from torch import optim
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable
import os
from tqdm import tqdm as tq
import time
import matplotlib.pyplot as plt
from .models import SUMNet, U_Net, R2U_Net, Discriminator
import json
from .data_loader import LungDataLoader
from .utils import dice_coefficient, plot_graphs, ch_shuffle

plt.switch_backend('agg')

def train_network(fold_no,save_path,json_path,datapath,lung_segpath,network,epochs=35,lrate=1e-4,adv=False):
    """Training function for SUMNet,UNet,R2Unet

    Parameters
    ----------
    fold_no: str
        Fold number on which training is to be performed
    save_path: str
        Folder location to save the models and other plots
    json_path: str
        Folder location at which json files are stored
    datapath: str
        Folder location of data
    lung_segpath: str
        Folder location at which lung segmentation files are stored
    network: str
        Network to be trained
    epochs: int, Default: 35
        Number of epochs for training
    lrate: int, Default= 1e-4
        Learnig rate

    Returns
    -------

    None
    """

    fold = 'fold'+str(fold_no)
    save_path = save_path+'/'+network+'/'+fold+'/'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    with open(json_path+fold+'_pos_neg_eq.json') as f:
        json_file = json.load(f)
        train_set = json_file['train_set']
        val_set = json_file['valid_set']


    trainDset = LungDataLoader(datapath=datapath,lung_path=lung_segpath,is_transform=True,json_file=json_file,split="train_set",img_size=512)
    valDset = LungDataLoader(datapath=datapath,lung_path=lung_segpath,is_transform=True,json_file=json_file,split="valid_set",img_size=512)
    trainDataLoader = data.DataLoader(trainDset,batch_size=4,shuffle=True,num_workers=4,pin_memory=True,drop_last=True)
    validDataLoader = data.DataLoader(valDset,batch_size=4,shuffle=False,num_workers=4,pin_memory=True,drop_last=True)

    if network == 'unet':
        net = U_Net(img_ch=1,output_ch=2)
    if network == 'r2unet':
        net = R2U_Net(img_ch=1,output_ch=2)
    if network == 'sumnet':
        net = SUMNet(in_ch=1,out_ch=2)

    use_gpu = torch.cuda.is_available()

    if use_gpu:
        net = net.cuda()

    optimizer = optim.Adam(net.parameters(), lr = lrate, weight_decay = 1e-5)
    if adv:
        netD2 = Discriminator(in_ch=2,out_ch=2)
        if use_gpu:
            netD2 = netD2.cuda()
        optimizerD2 = optim.Adam(netD2.parameters(), lr = 1e-4, weight_decay = 1e-5)
        criterionD = nn.BCELoss()
        D2_losses = []

    criterion = nn.BCEWithLogitsLoss()

    epochs = epochs
    trainLoss = []
    validLoss = []
    trainDiceCoeff_lungs = []
    validDiceCoeff_lungs = []
    start = time.time()

    bestValidDice = torch.zeros(1)
    bestValidDice_lungs = 0.0


    for epoch in range(epochs):
        epochStart = time.time()
        trainRunningLoss = 0
        validRunningLoss = 0
        trainBatches = 0
        validBatches = 0
        trainDice_lungs = 0
        validDice_lungs = 0


        net.train(True)

        for data1 in tq(trainDataLoader):
            img, mask = data1
            labels = mask
            if use_gpu:
                inputs = img.cuda()
                labels = labels.cuda()

            net_out = net(Variable(inputs))


            net_out_sf = F.softmax(net_out,dim=1)

            if adv:
                optimizerD2.zero_grad()
                # Concatenate real (GT) and fake (segmented) samples along dim 1
                d_in = torch.cat((net_out[:,1].unsqueeze(1),labels[:,1].unsqueeze(1).float()),dim=1)
                # Shuffling aling dim 1: {real,fake} OR {fake,real}
                d_in,shuffLabel = ch_shuffle(d_in)
                # D2 prediction
                confr = netD2(Variable(d_in)).view(d_in.size(0),-1)
                # Compute loss
                LD2 = criterionD(confr,shuffLabel.float().cuda())
                # Compute gradients
                LD2.backward()
                # Backpropagate
                optimizerD2.step()
                # Appending loss for each batch into the list
                D2_losses.append(LD2.item())
                optimizerD2.zero_grad()
                d2_in = torch.cat((net_out[:,1].unsqueeze(1),labels[:,1].unsqueeze(1).float()),dim=1)
                d2_in, d2_lb = ch_shuffle(d2_in)
                conffs2 = netD2(d2_in).view(d2_in.size(0),-1)
                LGadv2 = criterionD(conffs2,d2_lb.float().cuda()) # Aversarial loss 2

            BCE_Loss = criterion(net_out[:,1],labels[:,1])
            net_loss = BCE_Loss
            optimizer.zero_grad()
            net_loss.backward()
            optimizer.step()
            trainRunningLoss += net_loss.item()

            trainDice = dice_coefficient(net_out_sf,torch.argmax(labels,dim=1))
            trainDice_lungs += trainDice[0]

            trainBatches += 1
    #         if trainBatches>1:
    #             break

        trainLoss.append(trainRunningLoss/trainBatches)
        trainDiceCoeff_lungs.append(trainDice_lungs/trainBatches)

        with torch.no_grad():
            for data1 in tq(validDataLoader):

                imgs, mask = data1
                labels = mask
                if use_gpu:
                    inputs = imgs.cuda()
                    labels = labels.cuda()

                net_out = net(Variable(inputs))
                net_out_sf = F.softmax(net_out.data,dim=1)


                BCE_Loss = criterion(net_out[:,1],labels[:,1])
                net_loss = BCE_Loss

                val_dice = dice_coefficient(net_out_sf,torch.argmax(labels,dim=1))
                validDice_lungs += val_dice[0]
                validRunningLoss += net_loss.item()
                validBatches += 1
    #             if validBatches>1:
    #                 break

            validLoss.append(validRunningLoss/validBatches)
            validDiceCoeff_lungs.append(validDice_lungs/validBatches)

        if validDice_lungs.cpu() > bestValidDice_lungs:
            bestValidDice_lungs = validDice_lungs.cpu()
            torch.save(net.state_dict(), save_path+'sumnet_best_lungs.pt')
        
        plot_graphs(train_values=trainLoss, valid_values=validLoss,
        save_path=save_path, x_label='Epochs', y_label='Loss',
        plot_title='Running Loss', save_name='LossPlot.png')

        epochEnd = time.time()-epochStart
        print('Epoch: {:.0f}/{:.0f} | Train Loss: {:.5f} | Valid Loss: {:.5f}'
              .format(epoch+1, epochs, trainRunningLoss/trainBatches, validRunningLoss/validBatches))
        print('Dice | Train  | Lung {:.3f}  | Valid | Lung {:.3f} | '
              .format(trainDice_lungs/trainBatches, validDice_lungs/validBatches))

        print('\nTime: {:.0f}m {:.0f}s'.format(epochEnd//60,epochEnd%60))

        print('Saving losses')

        torch.save(trainLoss, save_path+'trainLoss.pt')
        torch.save(validLoss, save_path+'validLoss.pt')
        torch.save(trainDiceCoeff_lungs, save_path+'trainDice_lungs.pt')
        torch.save(validDiceCoeff_lungs, save_path+'validDice_lungs.pt')

    #     if epoch>1:
    #         break

    end = time.time()-start
    print('Training completed in {:.0f}m {:.0f}s'.format(end//60,end%60))

    plot_graphs(train_values=trainLoss, valid_values=validLoss,
    save_path=save_path, x_label='Epochs', y_label='Loss',
    plot_title='Loss plot', save_name='LossPlotFinal.png')

    plot_graphs(train_values=trainDiceCoeff_lungs, valid_values=validDiceCoeff_lungs,
    save_path=save_path, x_label='Epochs', y_label='Dice coefficient',
    plot_title='Dice coefficient', save_name='Dice_Plot.png')

