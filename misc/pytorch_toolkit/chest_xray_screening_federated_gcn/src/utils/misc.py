import numpy as np
from sklearn.metrics import cohen_kappa_score
import torch
import copy
def compute_edge_attr(A):
    edge=[]
    edge_attr=[]
    for j in range(0,14):
        for k in range(0,14):
            if j==k:
                continue
            edge.append(np.array([j,k]))
            edge_attr.append(A[j,k])
    
    edge=np.array(edge)
    edge_attr=np.array(edge_attr)
    
    edge=torch.from_numpy(np.transpose(edge))
    edge=edge.long()
    
    edge_attr=torch.from_numpy(edge_attr)
    edge_attr=torch.unsqueeze(edge_attr, dim=1)
    edge_attr=edge_attr.float()
       
    return edge, edge_attr

def compute_adjacency_matrix(adj_type, site, split_npz='/storage/aneesh/split.npz'):
    # load the npz file
    a=np.load(split_npz, allow_pickle=True)    
    gt=a['gt']
    clstr_assgn=a['clstr_assgn']
    trn_val_tst=a['trn_val_tst']
    del a
    
    if site==-999:
        idx=np.where(trn_val_tst==0)[0]
    else:
        idx=np.where((clstr_assgn==site) & (trn_val_tst==0))[0]
    gt=gt[idx]
    
    kappa=np.zeros((14,14))
    TP=np.zeros((14,14))
    TN=np.zeros((14,14))
    FP=np.zeros((14,14))
    FN=np.zeros((14,14))
    kappa=np.zeros((14,14))
    agree=np.zeros((14,14))
    
    for j in range(0,14):
        gt_j=gt[j]
        for k in range(0, 14):
            gt_k=gt[k]
            
            ## Kappa and agree are symmetric ie., A(i,j)=A(j,i)
            kappa[j,k]=cohen_kappa_score(gt_j, gt_k)
            agree[j,k]=(np.where(gt_j==gt_k)[0].shape[0])/gt.shape[0]
            
            # How many times are both j and k =1---> This will be symmetric
            TP[j,k]=(np.where((gt_j==1) & (gt_k==1))[0].shape[0])/gt.shape[0]
            # How many times are both j and k=0 ---> This will be symmetric
            TN[j,k]=(np.where((gt_j==0) & (gt_k==0))[0].shape[0])/gt.shape[0]
            
            ####### FP and FN will get reversed for A(i,j) and A(j,i)
            # How many time k is 1 but j is 0
            FP[j,k]=(np.where((gt_j==0) & (gt_k==1))[0].shape[0])/gt.shape[0]
            # How many time k is 0 but j is 1
            FN[j,k]=(np.where((gt_j==1) & (gt_k==0))[0].shape[0])/gt.shape[0]
            
    if adj_type=='kappa':
        A=kappa
    elif adj_type=='fraction_agreement':
        A=agree
    elif adj_type=='confusion_matrix':
        A=np.concatenate((np.expand_dims(TP, axis=2), np.expand_dims(TN, axis=2),
                          np.expand_dims(FP, axis=2), np.expand_dims(FN, axis=2)), axis=2)
                    
    if A.ndim==2:
        tmp_edge, edge_attr=compute_edge_attr(A)
    else:
        edge_lst=[]
        edge_attr_lst=[]
        for x in range(A.shape[2]):
            tmp_edge, tmp_edge_attr=compute_edge_attr(np.squeeze(A[:,:,x]))
            edge_lst.append(tmp_edge)
            edge_attr_lst.append(tmp_edge_attr)
        edge_attr=torch.cat(edge_attr_lst, dim=1)
    return tmp_edge, edge_attr

################ Compute weighted average of model weights ##################
def average_weights(w, cmb_wt, device):

    cmb_wt=np.array(cmb_wt)
    cmb_wt=cmb_wt.astype(np.float)
    cmb_wt=cmb_wt/np.sum(cmb_wt)
    wts = torch.tensor(cmb_wt).to(device)
    wts=wts.float()
    w_avg = copy.deepcopy(w[0])
    
    for key in w_avg.keys(): # for each layer
        layer = key.split('.')[-1]
        if layer == 'num_batches_tracked':
            for i in range(1,len(w)): # for each model
                w_avg[key] += w[i][key].to(device)
            w_avg[key] = torch.div(w_avg[key].float(), torch.tensor(len(w)).float()).to(torch.int64)
        else:
            w_avg[key]=torch.mul(w_avg[key].to(device), wts[0].to(float))
            for i in range(1,len(w)):
                w_avg[key] += torch.mul(w[i][key].to(device), wts[i].to(float))
    return w_avg

def aggregate_local_weights(wt0, wt1, wt2, wt3, wt4, device):
    
    wt=average_weights([wt0, wt1, wt2, wt3, wt4], 
                       [1.0, 2030.0/1997, 2093.0/1997, 1978.0/1997, 2122.0/1997],device)
    return wt

def compute_lcl_wt(epoch, cmb_wts, glbl_wt, prev_lcl_wt, device):
    
    cmb_wt=cmb_wts[epoch]
    lcl_wt=1-cmb_wt
    wt=average_weights([prev_lcl_wt, glbl_wt], [lcl_wt, cmb_wt],device)
    return wt
    