import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import argparse
from network.models import UNet
from train_utils.dataloader import CustomDataset
from train_utils.loss_functions import diceCoeff


def inference(tst_loader, model):
    dc_lst=[]
    with torch.no_grad():
        for i, data in enumerate(tst_loader):
            img = Variable(data['image'].float().to(device))
            mask = Variable(data['mask'].float().to(device))
            mask_pred = model(img)
            mask_pred = torch.sigmoid(mask_pred)
            
            dice_coeff = diceCoeff(mask_pred, mask, reduce=True)
            dc_lst.append(dice_coeff.data.cpu().numpy())

            mask_pred = mask_pred.data.cpu().numpy()[0][0]*255
            mask_pred = mask_pred.astype(np.uint8)

            print('Processed Image '+str(i)+'... , Dice='+str(dice_coeff))
    
    dc_lst=np.array(dc_lst)
    return dc_lst


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, required=False, help='Batch size..')
    parser.add_argument('--num_workers', type=int, default=1, required=False, help='Number of workers..')
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--saved_model', type=str, default='', required=True, help='path to the saved model weights')
    parser.add_argument('--val_data', type=str, default='', required=True, help='location of npy file of validation dataset')
    parser.add_argument('--out_path', type=str, default='', required=True, help='location where the output masks for the mass regions will be saved')
    args = parser.parse_args()

    batch_sz = args.batch_size
    num_workers = args.num_workers
    gpu = args.gpu
    device = 'cuda' if gpu else 'cpu'

    saved_model_wt=args.saved_model # path to save the model weight 
    val_data_pth=args.val_data #this will be a npy file which contains the validation data.
    out_pth=args.out_path
    
    
    #Prepare the Test Dataloader
    x_tst = np.load(val_data_pth, allow_pickle=True)
    tst_data = CustomDataset(x_tst, transform=None)
    tst_loader = DataLoader(tst_data, batch_size=batch_sz, shuffle=False, num_workers=num_workers)
    
    #Load the CNN model
    model = UNet(num_filters=32) # instantiate model

    # load model weights
    checkpoint = torch.load(saved_model_wt, map_location=torch.device(device))
    model.load_state_dict(checkpoint['state_dict'])
    del checkpoint 
    model.to(device)
    model.eval() # inference mode

    dc_lst=inference(tst_loader, model)
    print('Mean Dice: '+str(np.mean(dc_lst)))
    
    


