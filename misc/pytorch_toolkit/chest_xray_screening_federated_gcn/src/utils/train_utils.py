import torch
from torch import nn
from .model import First_Conv, Fully_Connected_Layer, GNN_Network
from .dataloader import construct_dataset
from .loss import Custom_Loss
from torch.utils.data import DataLoader
from torchvision import models
from .misc import compute_adjacency_matrix
import copy
from torch_geometric.data import Data as Data_GNN
from torch_geometric.data import DataLoader as DataLoader_GNN

# Train 1 batch update
def train_one_batch(sample, cnv_lyr, backbone_model, fc_layers, gnn_model, optim1, optim2, optim3, optim4,
                    trn_typ, criterion, device, edge_index=None, edge_attr=None):
    ##Keep gnn_model and optim4 as None if training is to be done without GNN
    img=sample['img']
    gt=sample['gt']

    img=img.to(device)
    gt=gt.to(device)

    ########Forward Pass ##############
    img_3chnl=cnv_lyr(img)
    gap_ftr=backbone_model(img_3chnl)
    ftr_lst, prd=fc_layers(gap_ftr)
    if gnn_model is not None:
        ftr_lst=torch.cat(ftr_lst, dim=1)

        data_lst=[]
        for k in range(0, ftr_lst.shape[0]):
            data_lst.append(Data_GNN(x=ftr_lst[k,:,:], edge_index=edge_index,
                                     edge_attr=edge_attr, y=torch.unsqueeze(gt[k,:], dim=1)))
        loader = DataLoader_GNN(data_lst, batch_size=ftr_lst.shape[0])
        loader=next(iter(loader)).to(device)
        gt=loader.y

        prd_final=gnn_model(loader)
    else:
        prd_final=prd

    loss=criterion(prd_final, gt)

    ####### Backward Pass ##########
    ### Remove previous gradients
    optim1.zero_grad()
    optim2.zero_grad()
    optim3.zero_grad()
    if optim4 is not None:
        optim4.zero_grad()

    ### Compute Gradients
    loss.backward()

    ### Optimizer Gradients
    #if training is without gnn
    if gnn_model is None:
        optim1.step()
        optim2.step()
        optim3.step()
        return cnv_lyr, backbone_model, fc_layers, loss, optim1, optim2, optim3
    #if training is with gnn
    if trn_typ=='full':
        optim1.step()
        optim2.step()

    optim3.step()
    optim4.step()
    return cnv_lyr, backbone_model, fc_layers, gnn_model, loss,  optim1, optim2, optim3, optim4

#### Train main
def train_end_to_end(lr, cnv_lyr, backbone_model, fc_layers, gnn_model,
                     train_loader, trn_typ, n_batches, criterion, device,
                     edge_index=None, edge_attr=None):

    cnv_lyr.train()
    backbone_model.train()
    fc_layers.train()

    # optimizer
    optim1 = torch.optim.Adam(cnv_lyr.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    optim2 = torch.optim.Adam(backbone_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    optim3 = torch.optim.Adam(fc_layers.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    if gnn_model is not None:
        gnn_model.train()
        optim4 = torch.optim.Adam(gnn_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)

    cnt=0
    trn_run_loss=0
    for i, sample in enumerate(train_loader):
        cnt=cnt+1
        if gnn_model is not None:
            cnv_lyr, backbone_model, fc_layers, gnn_model, loss,  optim1, optim2, optim3, optim4=train_one_batch(
                            sample, cnv_lyr, backbone_model, fc_layers, gnn_model, optim1, optim2, optim3, optim4,
                            trn_typ, criterion, device, edge_index, edge_attr)
        else:
            #Set gnn_model and optim4 as None if training is to be done without GNN
            cnv_lyr, backbone_model, fc_layers, loss, optim1, optim2, optim3=train_one_batch(
                            sample, cnv_lyr, backbone_model, fc_layers, None, optim1, optim2, optim3, None,
                            trn_typ, criterion, device)
        trn_run_loss=trn_run_loss+loss

        if (i+1) % 20== 0: # displays after every 20 batch updates
            print ("cnt {}, Train Loss: {:.4f}".format(cnt,(trn_run_loss/(cnt))), end ="\r")

        ############# Monitor Validation Acc and Early Stopping ############
        if cnt>=n_batches:
            break
    if gnn_model is None:
        return cnv_lyr, backbone_model, fc_layers

    return cnv_lyr, backbone_model, fc_layers, gnn_model

def initialize_training(site, img_pth, split_npz, train_transform, test_transform, b_sz, device):

    data_trn=construct_dataset(img_pth, split_npz, site, train_transform, tn_vl_idx=0)
    trn_loader=DataLoader(data_trn,b_sz, shuffle=True, num_workers=1, pin_memory=False, drop_last=True)

    data_val=construct_dataset(img_pth, split_npz, site, test_transform, tn_vl_idx=1)
    val_loader=DataLoader(data_val, 1, shuffle=False, num_workers=1, pin_memory=False, drop_last=True)

    criterion=Custom_Loss(site, device)
    edge_index, edge_attr= compute_adjacency_matrix('confusion_matrix', site, split_npz)

    return trn_loader, val_loader, criterion, edge_index, edge_attr

def initialize_model_weights(cnv_lyr, backbone_model, fc_layers, gnn_model):

    cnv_wt=copy.deepcopy(cnv_lyr.state_dict())
    backbone_wt=copy.deepcopy(backbone_model.state_dict())
    fc_wt=copy.deepcopy(fc_layers.state_dict())
    gnn_wt=copy.deepcopy(gnn_model.state_dict())

    return cnv_wt, backbone_wt, fc_wt, gnn_wt

def instantiate_architecture(ftr_dim, model_name, gnn=False):
    # If gnn=True, then instantiate the GNN architecture
    if model_name=='densenet':
        inp_dim=1024
        backbone_model=models.densenet121(pretrained=True)
        backbone_model.classifier=nn.Identity()
    elif model_name=='resnet':
        inp_dim=512
        backbone_model=models.resnet18(pretrained=True)
        backbone_model.fc=nn.Identity()

    elif model_name=='xception':
        inp_dim=2048
        backbone_model=xception.xception(pretrained=True)
        backbone_model.fc=nn.Identity()

    cnv_lyr=First_Conv()
    fc_layers=Fully_Connected_Layer(inp_dim, ftr_dim)
    if gnn:
        gnn_model=GNN_Network(in_chnls=512, base_chnls=1, grwth_rate=1, depth=1, aggr_md='mean', ftr_dim=4)
        return cnv_lyr, backbone_model, fc_layers, gnn_model

    return cnv_lyr, backbone_model, fc_layers
