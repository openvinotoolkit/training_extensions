import torch
from torch.utils import data
import torch.nn.functional as F
from torchvision import transforms
import os
import numpy as np
from tqdm import tqdm as tq
import matplotlib.pyplot as plt
import json
from .models import SUMNet, U_Net, R2U_Net
from .data_loader import LungDataLoader
from .utils import dice_coefficient, load_inference_model, load_checkpoint

plt.switch_backend('agg')
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def infer_lungseg(config, run_type='pytorch'):
    """ Inference script for lung segmentation

    Parameters
    ----------
    fold_no: int
        Fold number to which action is to be performed
    save_path: str
        Folder location to save the results
    network: str
        Network name
    jsonpath:
        Folder location where file is to be stored

    Returns
    -------
    None

    """
    fold_no = config["fold_no"]
    save_path = config["save_path"]
    network = config["network"]
    print(network)
    jsonpath = config["json_path"]
    datapath = config["data_path"]
    lung_segpath = config["lung_segpath"]
    fold = 'fold'+str(fold_no)
    save_path = os.path.join(save_path,network,fold)

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    with open(jsonpath) as f:
        json_file = json.load(f)
    testDset = LungDataLoader(datapath=datapath,lung_path = lung_segpath,is_transform=True,json_file=json_file,split="valid_set",img_size=512)
    testDataLoader = data.DataLoader(testDset,batch_size=1,shuffle=False,num_workers=4,pin_memory=True,drop_last=True)

    testBatches = 0
    testDice_lungs = 0
    dice_list = []
    use_gpu = torch.cuda.is_available()

    if run_type == 'pytorch':
        if network == 'sumnet':
            net = SUMNet(in_ch=1,out_ch=2)
        elif network == 'unet':
            net = U_Net(img_ch=1,output_ch=2)
        else:
            net = R2U_Net(img_ch=1,output_ch=2)
        if use_gpu:
            net = net.cuda()
        net = load_checkpoint(net,save_path+network+'_best_lungs.pt')

    elif run_type == 'onnx':
        net = load_inference_model(config,run_type='onnx')
    else:
        net = load_inference_model(config,run_type='ir')

    for data1 in tq(testDataLoader):
        inputs, labels = data1
        to_tensor = transforms.ToTensor()
        if run_type == 'pytorch':
            if use_gpu:
                inputs = inputs.cuda()

            net_out = net(inputs)
            net_out_sf = F.softmax(net_out.data,dim=1).detach().cpu()
        elif run_type == 'ir':
            net_out = net.infer(inputs={'input': inputs})['output']
            net_out = torch.tensor(net_out)
            net_out_sf = F.softmax(net_out.data,dim=0)
        else:
            ort_inputs = {net.get_inputs()[0].name: to_numpy(inputs)}
            net_out = net.run(None, ort_inputs)
            net_out = np.array(net_out)
            net_out = to_tensor(net_out).squeeze(1).transpose(dim0=1, dim1=0)
            net_out_sf = F.softmax(net_out.data,dim=1)

        test_dice = dice_coefficient(net_out_sf,torch.argmax(labels,dim=1))
        pred_max = torch.argmax(net_out_sf, dim=1)
        preds = torch.zeros(pred_max.shape)
        preds[pred_max == 1] = 1

        if not os.path.isdir(save_path+'seg_results/GT/'):
            os.makedirs(save_path+'seg_results/GT/')
            np.save(save_path+'seg_results/GT/image'+str(testBatches),labels[:,1].cpu())
        else:
            np.save(save_path+'seg_results/GT/image'+str(testBatches),labels[:,1].cpu())

        if not os.path.isdir(save_path+'seg_results/pred/'):
            os.makedirs(save_path+'seg_results/pred/')
            np.save(save_path+'seg_results/pred/image'+str(testBatches),preds.cpu())
        else:
            np.save(save_path+'seg_results/pred/image'+str(testBatches),preds.cpu())

        if not os.path.isdir(save_path+'seg_results/image/'):
            os.makedirs(save_path+'seg_results/image/')
            np.save(save_path+'seg_results/image/image'+str(testBatches),inputs.cpu())
        else:
            np.save(save_path+'seg_results/image/image'+str(testBatches),inputs.cpu())

        testDice_lungs += test_dice[0]
        dice_list.append(test_dice[0].cpu())
        testBatches += 1
    #     if testBatches>1:
    #         break

    dice = np.mean(dice_list)
    print("Result:",fold,dice)

    #Plots distribution of min values per volume
    plt.figure()
    plt.title('Distribution of Dice values')
    plt.hist(dice_list)
    plt.xlabel('Dice score')
    plt.ylabel('No. of Slices')
    plt.savefig(save_path+'dice_dist.jpg')
    # plt.show()
    plt.close()

    return dice



def visualise_seg(loadpath):
    """
    To visualise the segmentation performance(Qualitative results)

    Parameters
    ----------

    loadpath: str
        Folder location from where the files are to be loaded

    Returns
    -------
    None

    """

    image_list = os.listdir(loadpath+'GT/')
    count = 0
    for i in tq(image_list):
        img = np.load(loadpath+'image/'+i)
        GT = np.load(loadpath+'GT/'+i)
        pred = np.load(loadpath+'pred/'+i)

        plt.figure(figsize = [15,5])
        plt.subplot(141)
        plt.axis('off')
        plt.title('Input Image')
        plt.imshow(img[0][0],cmap = 'gray')
        plt.subplot(142)
        plt.axis('off')
        plt.title('GT')
        plt.imshow(GT[0],cmap = 'gray')
        plt.subplot(143)
        plt.axis('off')
        plt.title('Pred')
        plt.imshow(pred[0],cmap = 'gray')
        plt.subplot(144)
        plt.title('GT - Pred')
        plt.axis('off')
        test = GT[0]-pred[0]
        test[test>0] = 1
        test[test<=0] = 0
        plt.imshow(test,cmap = 'gray')
        count += 1

        if not os.path.isdir(loadpath+'seg_results/op_images/'):
            os.makedirs(loadpath+'seg_results/op_images/')
            plt.savefig(loadpath+'seg_results/op_images/img'+str(count)+'.jpg')
        else:
            plt.savefig(loadpath+'seg_results/op_images/img'+str(count)+'.jpg')
