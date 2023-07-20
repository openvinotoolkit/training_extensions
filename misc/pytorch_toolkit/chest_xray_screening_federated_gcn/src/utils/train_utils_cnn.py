import torch
import copy
from .inference_utils import inference
from .train_utils import train_end_to_end, instantiate_architecture
from .dataloader import construct_dataset
from torch.utils.data import DataLoader
from .loss import Custom_Loss
from .misc import compute_lcl_wt, aggregate_local_weights
from .transformations import train_transform, test_transform

def lcl_train(lr, trn_loader, val_loader, criterion, cnv_lyr1, backbone_model,fc_layers, device):
    n_batches = 4000
    ###### Compute the Validation accuracy #######
    prev_val=inference(cnv_lyr1, backbone_model, fc_layers, None, val_loader, criterion, device)
    ######## Train the entire network in an end-to-end manner ###
    train_end_to_end(lr, cnv_lyr1, backbone_model, fc_layers, None, trn_loader, None, n_batches, criterion, device)

    cnv_lyr1_wt=copy.deepcopy(cnv_lyr1.state_dict())
    backbone_wt=copy.deepcopy(backbone_model.state_dict())
    fc_wt=copy.deepcopy(fc_layers.state_dict())
    return prev_val, cnv_lyr1_wt,backbone_wt, fc_wt

def trainer_without_GNN( avg_schedule, lr, b_sz, img_pth, split_npz, train_transforms,
                         test_transforms, max_epochs, backbone, device, checkpoint='', savepath=''):

    cnv_lyr1, backbone_model, fc_layers = instantiate_architecture(ftr_dim=512, model_name=backbone)
    cnv_lyr1 = cnv_lyr1.to(device)
    backbone_model = backbone_model.to(device)
    fc_layers = fc_layers.to(device)
    if checkpoint!='':
        checkpoint=torch.load(checkpoint)
        cnv_wt=checkpoint['cnv_lyr_state_dict']
        backbone_wt=checkpoint['backbone_model_state_dict']
        fc_wt=checkpoint['fc_layers_state_dict']
        cnv_lyr1.load_state_dict(cnv_wt)
        backbone_model.load_state_dict(backbone_wt)
        fc_layers.load_state_dict(fc_wt)

    ### Dataloaders and model weights for each site
    # Site-0
    data_trn0=construct_dataset(img_pth, split_npz, site=0, transforms=train_transforms, tn_vl_idx=0)
    trn_loader0=DataLoader(data_trn0,b_sz, shuffle=True, num_workers=1, pin_memory=False, drop_last=True)
    data_val0=construct_dataset(img_pth, split_npz, site=0, transforms=test_transforms, tn_vl_idx=1)
    val_loader0=DataLoader(data_val0, b_sz, shuffle=False, num_workers=1, pin_memory=False, drop_last=True)

    # Site-1
    data_trn1=construct_dataset(img_pth, split_npz, site=1, transforms=train_transforms, tn_vl_idx=0)
    trn_loader1=DataLoader(data_trn1,b_sz, shuffle=True, num_workers=1, pin_memory=False, drop_last=True)
    data_val1=construct_dataset(img_pth, split_npz, site=1, transforms=test_transforms, tn_vl_idx=1)
    val_loader1=DataLoader(data_val1, b_sz, shuffle=False, num_workers=1, pin_memory=False, drop_last=True)

    # Site-2
    data_trn2=construct_dataset(img_pth, split_npz, site=2, transforms=train_transforms, tn_vl_idx=0)
    trn_loader2=DataLoader(data_trn2,b_sz, shuffle=True, num_workers=1, pin_memory=False, drop_last=True)
    data_val2=construct_dataset(img_pth, split_npz, site=2, transforms=test_transforms, tn_vl_idx=1)
    val_loader2=DataLoader(data_val2, b_sz, shuffle=False, num_workers=1, pin_memory=False, drop_last=True)

    # Site-3
    data_trn3=construct_dataset(img_pth, split_npz, site=3, transforms=train_transforms, tn_vl_idx=0)
    trn_loader3=DataLoader(data_trn3,b_sz, shuffle=True, num_workers=1, pin_memory=False, drop_last=True)
    data_val3=construct_dataset(img_pth, split_npz, site=3, transforms=test_transforms, tn_vl_idx=1)
    val_loader3=DataLoader(data_val3, b_sz, shuffle=False, num_workers=1, pin_memory=False, drop_last=True)


    # Site-4
    data_trn4=construct_dataset(img_pth, split_npz, site=4, transforms=train_transforms, tn_vl_idx=0)
    trn_loader4=DataLoader(data_trn4,b_sz, shuffle=True, num_workers=1, pin_memory=False, drop_last=True)
    data_val4=construct_dataset(img_pth, split_npz, site=4, transforms=test_transforms, tn_vl_idx=1)
    val_loader4=DataLoader(data_val4, b_sz, shuffle=False, num_workers=1, pin_memory=False, drop_last=True)


    criterion = Custom_Loss(site=-999,device=device)


    ###### Initialize model weights with pre-trained weights
    ## Global
    glbl_cnv_lyr1_wt=copy.deepcopy(cnv_lyr1.state_dict())
    glbl_backbone_wt=copy.deepcopy(backbone_model.state_dict())
    glbl_fc_wt=copy.deepcopy(fc_layers.state_dict())

    ## Site0
    sit0_cnv_lyr1_wt=copy.deepcopy(cnv_lyr1.state_dict())
    sit0_backbone_wt=copy.deepcopy(backbone_model.state_dict())
    sit0_fc_wt=copy.deepcopy(fc_layers.state_dict())

    ## Site 1
    sit1_cnv_lyr1_wt=copy.deepcopy(cnv_lyr1.state_dict())
    sit1_backbone_wt=copy.deepcopy(backbone_model.state_dict())
    sit1_fc_wt=copy.deepcopy(fc_layers.state_dict())

    ## Site 2
    sit2_cnv_lyr1_wt=copy.deepcopy(cnv_lyr1.state_dict())
    sit2_backbone_wt=copy.deepcopy(backbone_model.state_dict())
    sit2_fc_wt=copy.deepcopy(fc_layers.state_dict())

    ## Site 3
    sit3_cnv_lyr1_wt=copy.deepcopy(cnv_lyr1.state_dict())
    sit3_backbone_wt=copy.deepcopy(backbone_model.state_dict())
    sit3_fc_wt=copy.deepcopy(fc_layers.state_dict())

    ## Site 4
    sit4_cnv_lyr1_wt=copy.deepcopy(cnv_lyr1.state_dict())
    sit4_backbone_wt=copy.deepcopy(backbone_model.state_dict())
    sit4_fc_wt=copy.deepcopy(fc_layers.state_dict())

    ###### Now begin training
    max_val=0
    for epoch in range(0, max_epochs):

        ############ Perform the local trainings for each site #####
        ## Site 0
        print('\n \n SITE 0 \n')
        tmp_cnv_lyr1_wt=compute_lcl_wt(epoch, avg_schedule, glbl_cnv_lyr1_wt, sit0_cnv_lyr1_wt, device)
        tmp_backbone_wt=compute_lcl_wt(epoch, avg_schedule, glbl_backbone_wt, sit0_backbone_wt, device)
        tmp_fc_wt=compute_lcl_wt(epoch, avg_schedule, glbl_fc_wt, sit0_fc_wt, device)
        # Load the weights
        cnv_lyr1.load_state_dict(tmp_cnv_lyr1_wt)
        backbone_model.load_state_dict(tmp_backbone_wt)
        fc_layers.load_state_dict(tmp_fc_wt)

        prev_val0, sit0_cnv_lyr1_wt,sit0_backbone_wt, sit0_fc_wt=lcl_train(lr, trn_loader0, val_loader0, criterion,
                                                               cnv_lyr1, backbone_model,fc_layers, device )

        del tmp_cnv_lyr1_wt, tmp_backbone_wt, tmp_fc_wt


        ## Site 1
        print('\n \n SITE 1 \n')
        tmp_cnv_lyr1_wt=compute_lcl_wt(epoch, avg_schedule, glbl_cnv_lyr1_wt, sit1_cnv_lyr1_wt, device)
        tmp_backbone_wt=compute_lcl_wt(epoch, avg_schedule, glbl_backbone_wt, sit1_backbone_wt, device)
        tmp_fc_wt=compute_lcl_wt(epoch, avg_schedule, glbl_fc_wt, sit1_fc_wt, device)
        # Load the weights
        cnv_lyr1.load_state_dict(tmp_cnv_lyr1_wt)
        backbone_model.load_state_dict(tmp_backbone_wt)
        fc_layers.load_state_dict(tmp_fc_wt)

        prev_val1, sit1_cnv_lyr1_wt,sit1_backbone_wt, sit1_fc_wt=lcl_train(lr, trn_loader1, val_loader1, criterion,
                                                                cnv_lyr1, backbone_model,fc_layers, device )

        del tmp_cnv_lyr1_wt, tmp_backbone_wt, tmp_fc_wt


        ## Site 2
        print('\n \n SITE 2 \n')
        tmp_cnv_lyr1_wt=compute_lcl_wt(epoch, avg_schedule, glbl_cnv_lyr1_wt, sit2_cnv_lyr1_wt, device)
        tmp_backbone_wt=compute_lcl_wt(epoch, avg_schedule, glbl_backbone_wt, sit2_backbone_wt, device)
        tmp_fc_wt=compute_lcl_wt(epoch, avg_schedule, glbl_fc_wt, sit2_fc_wt, device)
        # Load the weights
        cnv_lyr1.load_state_dict(tmp_cnv_lyr1_wt)
        backbone_model.load_state_dict(tmp_backbone_wt)
        fc_layers.load_state_dict(tmp_fc_wt)

        prev_val2, sit2_cnv_lyr1_wt,sit2_backbone_wt, sit2_fc_wt=lcl_train(lr, trn_loader2, val_loader2, criterion,
                                                                cnv_lyr1, backbone_model,fc_layers, device )

        del tmp_cnv_lyr1_wt, tmp_backbone_wt, tmp_fc_wt


        ## Site 3
        print('\n \n SITE 3 \n')
        tmp_cnv_lyr1_wt=compute_lcl_wt(epoch, avg_schedule, glbl_cnv_lyr1_wt, sit3_cnv_lyr1_wt, device)
        tmp_backbone_wt=compute_lcl_wt(epoch, avg_schedule, glbl_backbone_wt, sit3_backbone_wt, device)
        tmp_fc_wt=compute_lcl_wt(epoch, avg_schedule, glbl_fc_wt, sit3_fc_wt, device)
        # Load the weights
        cnv_lyr1.load_state_dict(tmp_cnv_lyr1_wt)
        backbone_model.load_state_dict(tmp_backbone_wt)
        fc_layers.load_state_dict(tmp_fc_wt)

        prev_val3, sit3_cnv_lyr1_wt,sit3_backbone_wt, sit3_fc_wt=lcl_train(lr, trn_loader3, val_loader3, criterion,
                                                                cnv_lyr1, backbone_model,fc_layers , device)

        del tmp_cnv_lyr1_wt, tmp_backbone_wt, tmp_fc_wt


        ## Site 4
        print('\n \n SITE 4 \n')
        tmp_cnv_lyr1_wt=compute_lcl_wt(epoch, avg_schedule, glbl_cnv_lyr1_wt, sit4_cnv_lyr1_wt, device)
        tmp_backbone_wt=compute_lcl_wt(epoch, avg_schedule, glbl_backbone_wt, sit4_backbone_wt, device)
        tmp_fc_wt=compute_lcl_wt(epoch, avg_schedule, glbl_fc_wt, sit4_fc_wt, device)
        # Load the weights
        cnv_lyr1.load_state_dict(tmp_cnv_lyr1_wt)
        backbone_model.load_state_dict(tmp_backbone_wt)
        fc_layers.load_state_dict(tmp_fc_wt)

        prev_val4, sit4_cnv_lyr1_wt,sit4_backbone_wt, sit4_fc_wt=lcl_train(lr, trn_loader4, val_loader4, criterion,
                                                                cnv_lyr1, backbone_model,fc_layers , device)

        del tmp_cnv_lyr1_wt, tmp_backbone_wt, tmp_fc_wt


        avg_val=(prev_val0+prev_val1+prev_val2+prev_val3+prev_val4)/5

        if avg_val>max_val:
            max_val=avg_val
            # save model weight, local weights
            torch.save({
                    'cnv_lyr1_state_dict': glbl_cnv_lyr1_wt,
                    'backbone_model_state_dict': glbl_backbone_wt,
                    'fc_layers_state_dict': glbl_fc_wt,
                    }, savepath+'best_glbl_weights.pth')

            torch.save({
                    'cnv_lyr1_state_dict': sit0_cnv_lyr1_wt,
                    'backbone_model_state_dict': sit0_backbone_wt,
                    'fc_layers_state_dict': sit0_fc_wt,
                    }, savepath+'best_site0_weights.pth')

            torch.save({
                    'cnv_lyr1_state_dict': sit1_cnv_lyr1_wt,
                    'backbone_model_state_dict': sit1_backbone_wt,
                    'fc_layers_state_dict': sit1_fc_wt,
                    }, savepath+'best_site1_weights.pth')

            torch.save({
                    'cnv_lyr1_state_dict': sit2_cnv_lyr1_wt,
                    'backbone_model_state_dict': sit2_backbone_wt,
                    'fc_layers_state_dict': sit2_fc_wt,
                    }, savepath+'best_site2_weights.pth')

            torch.save({
                    'cnv_lyr1_state_dict': sit3_cnv_lyr1_wt,
                    'backbone_model_state_dict': sit3_backbone_wt,
                    'fc_layers_state_dict': sit3_fc_wt,
                    }, savepath+'best_site3_weights.pth')

            torch.save({
                    'cnv_lyr1_state_dict': sit4_cnv_lyr1_wt,
                    'backbone_model_state_dict': sit4_backbone_wt,
                    'fc_layers_state_dict': sit4_fc_wt,
                    }, savepath+'best_site4_weights.pth')

        ######### aggregate to compute global weight ###############

        glbl_cnv_lyr1_wt=aggregate_local_weights(sit0_cnv_lyr1_wt, sit1_cnv_lyr1_wt, sit2_cnv_lyr1_wt,
                                                sit3_cnv_lyr1_wt, sit4_cnv_lyr1_wt, device)

        glbl_backbone_wt=aggregate_local_weights(sit0_backbone_wt, sit1_backbone_wt, sit2_backbone_wt,
                                                sit3_backbone_wt, sit4_backbone_wt, device)

        glbl_fc_wt=aggregate_local_weights(sit0_fc_wt, sit1_fc_wt, sit2_fc_wt, sit3_fc_wt, sit4_fc_wt, device)


    ###### Just before returning, save the final weights

    # save model weight, local weights
    torch.save({
            'cnv_lyr1_state_dict': glbl_cnv_lyr1_wt,
            'backbone_model_state_dict': glbl_backbone_wt,
            'fc_layers_state_dict': glbl_fc_wt,
            }, savepath+'final_glbl_weights.pth')

    torch.save({
            'cnv_lyr1_state_dict': sit0_cnv_lyr1_wt,
            'backbone_model_state_dict': sit0_backbone_wt,
            'fc_layers_state_dict': sit0_fc_wt,
            }, savepath+'final_site0_weights.pth')

    torch.save({
            'cnv_lyr1_state_dict': sit1_cnv_lyr1_wt,
            'backbone_model_state_dict': sit1_backbone_wt,
            'fc_layers_state_dict': sit1_fc_wt,
            }, savepath+'final_site1_weights.pth')

    torch.save({
            'cnv_lyr1_state_dict': sit2_cnv_lyr1_wt,
            'backbone_model_state_dict': sit2_backbone_wt,
            'fc_layers_state_dict': sit2_fc_wt,
            }, savepath+'final_site2_weights.pth')

    torch.save({
            'cnv_lyr1_state_dict': sit3_cnv_lyr1_wt,
            'backbone_model_state_dict': sit3_backbone_wt,
            'fc_layers_state_dict': sit3_fc_wt,
            }, savepath+'final_site3_weights.pth')

    torch.save({
            'cnv_lyr1_state_dict': sit4_cnv_lyr1_wt,
            'backbone_model_state_dict': sit4_backbone_wt,
            'fc_layers_state_dict': sit4_fc_wt,
            }, savepath+'final_site4_weights.pth')

def train_model(config):
    if torch.cuda.is_available() and config['gpu']=='True':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    avg_schedule = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    trainer_without_GNN(avg_schedule, config['lr'], config['batch_size'], config['data'],
                        config['split_npz'], train_transform, test_transform, config['epochs'],
                        config['backbone'], device, config['checkpoint'], config['savepath'])
