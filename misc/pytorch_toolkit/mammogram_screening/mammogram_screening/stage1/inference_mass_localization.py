import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import onnxruntime
from openvino.inference_engine import IECore
from torchvision import transforms
import os
from ..train_utils.models import UNet
from ..train_utils.dataloader import Stage1Dataset
from ..train_utils.loss_functions import diceCoeff
from ..train_utils.get_config import get_config

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

class InferenceStage1():
    def __init__(self, dataloader_test, checkpoint, device):
        self.dataloader = dataloader_test
        self.checkpoint = checkpoint
        self.device = device
    
    def load_model(self, type='pytorch'):
        if type == 'onnx':
            model = onnxruntime.InferenceSession(self.checkpoint)
        elif type == 'pytorch':
            model = UNet(num_filters=32) 
            checkpoint = torch.load(self.checkpoint, map_location=torch.device(self.device))
            model.load_state_dict(checkpoint['state_dict'])
            model.to(self.device)
            model.eval()
        else:
            ie = IECore()
            model_xml = os.path.splitext(self.checkpoint)[0] + ".xml"
            model_bin = os.path.splitext(model_xml)[0] + ".bin"
            model_temp = ie.read_network(model_xml, model_bin)
            model = ie.load_network(network=model_temp, device_name='CPU')
        return model

    def inference(self, model, runtype):
        dc_lst = []
        to_tensor = transforms.ToTensor()
        with torch.no_grad():
            for i, data in enumerate(self.dataloader):
                img = Variable(data['image'].float().to(self.device))
                mask = Variable(data['mask'].float().to(self.device))
                if runtype == 'pytorch':
                    # print(img.shape)
                    out = model(img)
                    out = torch.sigmoid(out)
                elif runtype == "onnx":
                    model = onnxruntime.InferenceSession(self.checkpoint)
                    ort_inputs = {model.get_inputs()[0].name: to_numpy(img)}
                    out = model.run(None, ort_inputs)
                    out = np.array(out)
                    out = out.squeeze(0).squeeze(1).squeeze(0)
                    out = to_tensor(out).to(self.device) #.transpose(dim0=1, dim1=0)

                else:
                    out = model.infer(inputs={'input': img})['output']
                    # print(out.shape)
                    out = out.squeeze(0).squeeze(0)
                    out = to_tensor(out)#.squeeze(1)

                dice_coeff = diceCoeff(out, mask, reduce=True)
                dc_lst.append(dice_coeff.data.cpu().numpy())

        dc_lst = np.array(dc_lst)
        dice_mean = np.mean(dc_lst)
        return dice_mean

if __name__ == '__main__':

    config = get_config('inference')
    batch_sz = config['batch_size']
    num_workers = config['num_workers']
    gpu = config['gpu']
    device = 'cuda' if gpu else 'cpu'
    val_data_pth = config['val_data_path']
    model_path = config['checkpoint']
    onnx_model_path = config['onnx_checkpoint']

    #Prepare the Test Dataloader
    x_tst = np.load(val_data_pth, allow_pickle=True)
    tst_data = Stage1Dataset(x_tst, transform=None)
    tst_loader = DataLoader(tst_data, batch_size=batch_sz, shuffle=False, num_workers=num_workers)

    inference = InferenceStage1(dataloader_test=tst_loader, checkpoint=model_path, device=device)
    model = inference.load_model(type='pytorch')
    mean_dice = inference.inference(model, runtype='pytorch')

    print(f'Mean Dice:{mean_dice}')

    # inference = InferenceStage1(dataloader_test=tst_loader, checkpoint=onnx_model_path, device=device)
    # model = inference.load_model(type='onnx')
    # mean_dice = inference.inference(model, runtype='onnx')

    # print(f'Mean Dice:{mean_dice}')

    # inference = InferenceStage1(dataloader_test=tst_loader, checkpoint=onnx_model_path, device='cpu')
    # model = inference.load_model(type='cpu')
    # mean_dice = inference.inference(model, runtype='cpu')

    # print(f'Mean Dice:{mean_dice}')
