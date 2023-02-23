import torch
from .train_utils import train_end_to_end, initialize_training
from .inference_utils import inference
from .train_utils import initialize_model_weights, instantiate_architecture
from .misc import aggregate_local_weights, save_model_weights
from .transformations import train_transform, test_transform
import copy

def lcl_train_gnn(lr, trn_loader, val_loader, criterion, cnv_lyr,
                  backbone_model,fc_layers, gnn_model, edge_index, edge_attr, device):

    n_batches=1500
    ####### Freeze and train the part which is specific to each site only
    print('Freeze global CNN, fine-tune GNN ...')
    cnv_lyr, backbone_model, fc_layers, gnn_model=train_end_to_end(lr, cnv_lyr, backbone_model,
                                                fc_layers, gnn_model, trn_loader,'gnn', n_batches, criterion, device,
                                                edge_index, edge_attr)

    ###### Compute the Validation accuracy #######
    print('Computing Validation Performance ...')
    prev_val=inference(cnv_lyr, backbone_model, fc_layers, gnn_model, val_loader, criterion, device,
                        edge_index, edge_attr)

    ######## Train the entire network in an end-to-end manner ###
    print('Train end-to-end for Local Site ...')
    cnv_lyr, backbone_model, fc_layers, gnn_model=train_end_to_end(lr, cnv_lyr, backbone_model, fc_layers,
                                                        gnn_model, trn_loader,'full', 2*n_batches, criterion, device,
                                                        edge_index, edge_attr)

    cnv_wt=copy.deepcopy(cnv_lyr.state_dict())
    backbone_wt=copy.deepcopy(backbone_model.state_dict())
    fc_wt=copy.deepcopy(fc_layers.state_dict())
    gnn_wt=copy.deepcopy(gnn_model.state_dict())

    return prev_val, cnv_wt,backbone_wt, fc_wt, gnn_wt

#Main function for training
def trainer_with_GNN(lr, b_sz, img_pth, split_npz, train_transform, test_transform,
                     max_epochs, backbone, device, restart_checkpoint='', savepoint=''):

    ###### Instantiate the CNN-GNN Architecture ##############
    cnv_lyr, backbone_model, fc_layers, gnn_model=instantiate_architecture(ftr_dim=512, model_name=backbone, gnn=True)
    cnv_lyr = cnv_lyr.to(device)
    fc_layers = fc_layers.to(device)
    backbone_model = backbone_model.to(device)
    gnn_model = gnn_model.to(device)

    ############## Initialize Data Loaders #################
    trn_loader0, val_loader0, criterion0, edge_index0, edge_attr0=initialize_training(0, img_pth, split_npz,
                                                              train_transform, test_transform, b_sz, device=device)

    trn_loader1, val_loader1, criterion1, edge_index1, edge_attr1=initialize_training(1, img_pth, split_npz,
                                                              train_transform, test_transform, b_sz, device=device)

    trn_loader2, val_loader2, criterion2, edge_index2, edge_attr2=initialize_training(2, img_pth, split_npz,
                                                              train_transform, test_transform, b_sz, device=device)

    trn_loader3, val_loader3, criterion3, edge_index3, edge_attr3=initialize_training(3, img_pth, split_npz,
                                                              train_transform, test_transform, b_sz, device=device)

    trn_loader4, val_loader4, criterion4, edge_index4, edge_attr4=initialize_training(4, img_pth, split_npz,
                                                              train_transform, test_transform, b_sz, device=device)

    ### Initialize local and global model weights with the Imagenet pre-trained weights for backbone
    #and identical model weights for the other layers.

    glbl_cnv_wt, glbl_backbone_wt, glbl_fc_wt, gnn_wt=initialize_model_weights(cnv_lyr, backbone_model,
                                                                                    fc_layers, gnn_model)
    sit0_gnn_wt=copy.deepcopy(gnn_wt)
    sit1_gnn_wt=copy.deepcopy(gnn_wt)
    sit2_gnn_wt=copy.deepcopy(gnn_wt)
    sit3_gnn_wt=copy.deepcopy(gnn_wt)
    sit4_gnn_wt=copy.deepcopy(gnn_wt)

    del gnn_wt
    # Load previous checkpoint if resuming the  training else comment out
    if restart_checkpoint!='':
        checkpoint=torch.load(restart_checkpoint)
        glbl_cnv_wt=checkpoint['cnv_lyr_state_dict']
        glbl_backbone_wt=checkpoint['backbone_model_state_dict']
        glbl_fc_wt=checkpoint['fc_layers_state_dict']
        sit0_gnn_wt=checkpoint['sit0_gnn_model']
        sit1_gnn_wt=checkpoint['sit1_gnn_model']
        sit2_gnn_wt=checkpoint['sit2_gnn_model']
        sit3_gnn_wt=checkpoint['sit3_gnn_model']
        sit4_gnn_wt=checkpoint['sit4_gnn_model']

    ################ Begin Actual Training ############
    max_val=0
    for epoch in range(0, max_epochs):
        print('############ Epoch: '+str(epoch)+'   #################')
        ###### Load the global model weights ########
        cnv_lyr.load_state_dict(glbl_cnv_wt)
        backbone_model.load_state_dict(glbl_backbone_wt)
        fc_layers.load_state_dict(glbl_fc_wt)
        gnn_model.load_state_dict(sit0_gnn_wt)

        print('\n \n SITE 0 \n')
        prv_val0, sit0_cnv_wt,sit0_backbone_wt, sit0_fc_wt, sit0_gnn_wt=lcl_train_gnn(lr, trn_loader0, val_loader0,
                                                criterion0, cnv_lyr, backbone_model,fc_layers, gnn_model,
                                                                                 edge_index0, edge_attr0, device)

        cnv_lyr.load_state_dict(glbl_cnv_wt)
        backbone_model.load_state_dict(glbl_backbone_wt)
        fc_layers.load_state_dict(glbl_fc_wt)
        gnn_model.load_state_dict(sit1_gnn_wt)

        print('\n \n SITE 1 \n')
        prv_val1, sit1_cnv_wt,sit1_backbone_wt, sit1_fc_wt, sit1_gnn_wt=lcl_train_gnn(lr, trn_loader1, val_loader1,
                                                criterion1, cnv_lyr, backbone_model,fc_layers, gnn_model,
                                                                                 edge_index1, edge_attr1, device)

        cnv_lyr.load_state_dict(glbl_cnv_wt)
        backbone_model.load_state_dict(glbl_backbone_wt)
        fc_layers.load_state_dict(glbl_fc_wt)
        gnn_model.load_state_dict(sit2_gnn_wt)

        print('\n \n SITE 2 \n')
        prv_val2, sit2_cnv_wt,sit2_backbone_wt, sit2_fc_wt, sit2_gnn_wt=lcl_train_gnn(lr, trn_loader2, val_loader2,
                                                criterion2, cnv_lyr, backbone_model,fc_layers, gnn_model,
                                                                                 edge_index2, edge_attr2, device)

        cnv_lyr.load_state_dict(glbl_cnv_wt)
        backbone_model.load_state_dict(glbl_backbone_wt)
        fc_layers.load_state_dict(glbl_fc_wt)
        gnn_model.load_state_dict(sit3_gnn_wt)

        print('\n \n SITE 3 \n')
        prv_val3, sit3_cnv_wt,sit3_backbone_wt, sit3_fc_wt, sit3_gnn_wt=lcl_train_gnn(lr, trn_loader3, val_loader3,
                                                criterion3, cnv_lyr, backbone_model,fc_layers, gnn_model,
                                                                                 edge_index3, edge_attr3, device)

        cnv_lyr.load_state_dict(glbl_cnv_wt)
        backbone_model.load_state_dict(glbl_backbone_wt)
        fc_layers.load_state_dict(glbl_fc_wt)
        gnn_model.load_state_dict(sit4_gnn_wt)

        print('\n \n SITE 4 \n')
        prv_val4, sit4_cnv_wt,sit4_backbone_wt, sit4_fc_wt, sit4_gnn_wt=lcl_train_gnn(lr, trn_loader4, val_loader4,
                                                criterion4, cnv_lyr, backbone_model,fc_layers, gnn_model,
                                                                                 edge_index4, edge_attr4, device)

        avg_val=(prv_val0+prv_val1+prv_val2+prv_val3+prv_val4)/5
        print('Avg Val AUC: '+str(avg_val))

        if avg_val>max_val:
            max_val=avg_val
            mx_nm=savepoint+'best_weight_'+str(max_val)+'_'+str(epoch)+'.pt'
            save_model_weights(mx_nm, glbl_cnv_wt, glbl_backbone_wt,
                               glbl_fc_wt, sit0_gnn_wt, sit1_gnn_wt,
                               sit2_gnn_wt, sit3_gnn_wt, sit4_gnn_wt)
            print('Validation Performance Improved !')

        ############### Compute the global model weights #############

        glbl_cnv_wt=aggregate_local_weights(sit0_cnv_wt, sit1_cnv_wt, sit2_cnv_wt,
                                                sit3_cnv_wt, sit4_cnv_wt, device)

        glbl_backbone_wt=aggregate_local_weights(sit0_backbone_wt, sit1_backbone_wt, sit2_backbone_wt,
                                                sit3_backbone_wt, sit4_backbone_wt, device)

        glbl_fc_wt=aggregate_local_weights(sit0_fc_wt, sit1_fc_wt, sit2_fc_wt, sit3_fc_wt, sit4_fc_wt, device)

def train_model(config):
    if torch.cuda.is_available() and config['gpu']=='True':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    trainer_with_GNN(config['lr'],config['batch_size'], config['data'], config['split_npz'],
                        train_transform, test_transform, config['epochs'], config['backbone'],
                        device, config['checkpoint'], config['savepath'] )
