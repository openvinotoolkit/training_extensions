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
import argparse

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

class InferenceStage1():
    def __init__(self, dataloader_test, checkpoint, device):
        self.dataloader = dataloader_test
        self.checkpoint = checkpoint
        self.device = device

    def load_model(self, run_type='pytorch'):
        if run_type == 'onnx':
            model = onnxruntime.InferenceSession(self.checkpoint)
        elif run_type == 'pytorch':
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
            for data in self.dataloader:
                img = Variable(data['image'].float().to(self.device))
                mask = Variable(data['mask'].float().to(self.device))
                if runtype == 'pytorch':
                    out = model(img)
                    out = torch.sigmoid(out)
                elif runtype == "onnx":
                    model = onnxruntime.InferenceSession(self.checkpoint)
                    ort_inputs = {model.get_inputs()[0].name: to_numpy(img)}
                    out = model.run(None, ort_inputs)
                    out = np.array(out)
                    out = out.squeeze(0).squeeze(1).squeeze(0)
                    out = to_tensor(out).to(self.device)

                else:
                    out = model.infer(inputs={'input': img})['output']
                    out = out.squeeze(0).squeeze(0)
                    out = to_tensor(out)

                dice_coeff = diceCoeff(out, mask, reduce=True)
                dc_lst.append(dice_coeff.data.cpu().numpy())

        dc_lst = np.array(dc_lst)
        dice_mean = np.mean(dc_lst)
        return dice_mean

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, help='absolute path to config files', type=str)
    parser.add_argument('--runtype', required=True, help="Specify runtype ['pytorch','onnx','cpu']", type=str)
    args = parser.parse_args()

    config = get_config(action='inference', config_path=args.path, stage='stage1')
    batch_sz = config['batch_size']
    num_workers = config['num_workers']
    val_data_pth = config['val_data_path']
    if args.runtype == 'pytorch':
        model_path = config['checkpoint']
        devicex = 'cuda'
    else:
        model_path = config['onnx_checkpoint']
        devicex = 'cpu'

    #Prepare the Test Dataloader
    x_tst = np.load(val_data_pth, allow_pickle=True)
    tst_data = Stage1Dataset(x_tst, transform=None)
    tst_loader = DataLoader(tst_data, batch_size=batch_sz, shuffle=False, num_workers=num_workers)

    inference = InferenceStage1(dataloader_test=tst_loader, checkpoint=model_path, device=devicex)
    model_x = inference.load_model(run_type=args.runtype)
    mean_dice = inference.inference(model_x, runtype=args.runtype)

    print(f'Mean Dice:{mean_dice}')
