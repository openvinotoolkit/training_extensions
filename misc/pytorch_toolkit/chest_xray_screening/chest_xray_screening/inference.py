import numpy as np
import argparse
import torch
from torch.backends import cudnn
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from math import sqrt
import json
import os
from .utils.dataloader import RSNADataSet
from .utils.score import compute_auroc
from .utils.model import DenseNet121, DenseNet121Eff, load_checkpoint
import onnx
import onnxruntime
from PIL import Image
from torchvision import transforms
from openvino.inference_engine import IECore

class RSNAInference():
    def __init__(self, model, data_loader_test, class_count, checkpoint, class_names, device):
        self.device = device
        self.model = model.to(self.device)
        self.checkpoint = checkpoint
        load_checkpoint(self.model, self.checkpoint)
        self.data_loader_test = data_loader_test
        self.class_count = class_count
        self.class_names = class_names
        self.model.eval()

    def test_onnx(self, img_path, onnx_checkpoint):
        onnx_model = onnx.load(onnx_checkpoint)
        onnx.checker.check_model(onnx_model)
        # The validity of the ONNX graph is verified by checking the model’s version,
        # the graph’s structure, as well as the nodes and their inputs and outputs.
        ort_session = onnxruntime.InferenceSession(onnx_checkpoint)
        sample_image = Image.open(img_path).convert('RGB')
        to_tensor = transforms.ToTensor()
        sample_image = to_tensor(sample_image)
        sample_image.unsqueeze_(0)
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(sample_image)}
        ort_outs = ort_session.run(None, ort_inputs)
        torch_out = self.model(sample_image.cuda())
        np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    def load_inference_model(self, run_type, onnx_checkpoint):
        if run_type == 'pytorch':
            model = self.model
        elif run_type == 'onnx':
            model = onnxruntime.InferenceSession(onnx_checkpoint)
        else:
            ie = IECore()
            model_xml = os.path.splitext(onnx_checkpoint)[0] + ".xml"
            model_bin = os.path.splitext(model_xml)[0] + ".bin"
            model_temp = ie.read_network(model_xml, model_bin)
            model = ie.load_network(network=model_temp, device_name='CPU')
        return model

    def validate_models(self, run_type, onnx_checkpoint =''):
        cudnn.benchmark = True
        out_gt = torch.FloatTensor()
        out_pred = torch.FloatTensor()
        model = self.load_inference_model(run_type, onnx_checkpoint)

        with torch.no_grad():
            for var_input, var_target in self.data_loader_test:
                if run_type in ('pytorch', 'onnx'):
                    var_target = var_target.to(self.device)
                    var_input = var_input.to(self.device)
                    out_gt = out_gt.to(self.device)
                    out_pred = out_pred.to(self.device)
                    out_gt = torch.cat((out_gt, var_target), 0)
                else:
                    out_gt = torch.cat((out_gt, var_target), 0)
                _, c, h, w = var_input.size()
                var_input = var_input.view(-1, c, h, w)
                to_tensor = transforms.ToTensor()
                if run_type == 'pytorch':
                    out = model(var_input)
                elif run_type == 'onnx':
                    ort_inputs = {model.get_inputs()[0].name: to_numpy(var_input)}
                    out = model.run(None, ort_inputs)
                    out = np.array(out)
                    out = to_tensor(out).squeeze(1).transpose(dim0=1, dim1=0).to(self.device)
                else:
                    out = model.infer(inputs={'input': var_input})['output']
                    out = to_tensor(out).squeeze(1)

                out_pred = torch.cat((out_pred, out), 0)
                one_hot_gt = tfunc.one_hot(out_gt.squeeze(1).long()).float()
                auroc_individual = compute_auroc(one_hot_gt, out_pred, self.class_count)
        auroc_mean = np.array(auroc_individual).mean()
        return auroc_mean


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()



def main(args):
    checkpoint= args.checkpoint
    class_count = 3

    class_names = ['Lung Opacity', 'Normal', 'No Lung Opacity / Not Normal']
    dpath = args.dpath
    img_pth = os.path.join(args.dpath, 'processed_data/')
    numpy_path = os.path.join(args.dpath, 'data_split/')
    with open(os.path.join(dpath, 'rsna_annotation.json')) as lab_file:
        labels = json.load(lab_file)
    test_list = np.load(os.path.join(numpy_path, 'test_list.npy')).tolist()

    dataset_test = RSNADataSet(test_list, labels, img_pth, transform=True)
    data_loader_test = DataLoader(
        dataset=dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=False)

    if args.optimised:
        alpha = args.alpha
        phi = args.phi
        beta = args.beta
        if beta is None:
            beta = round(sqrt(2 / alpha), 3)
        alpha = alpha ** phi
        beta = beta ** phi
        model = DenseNet121Eff(alpha, beta, class_count)
    else:
        model = DenseNet121(class_count)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use gpu if available

    rsna_inference = RSNAInference(model, data_loader_test, class_count, checkpoint, class_names, device)

    test_auroc = rsna_inference.validate_models(run_type='pytorch')
    print(f"Test AUROC is {test_auroc}")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
        required=False,
        help="start training from a checkpoint model weight",
        default= None,
        type = str)
    parser.add_argument("--dpath",
        required=True,
        help="Path to dataset",
        type =str)
    parser.add_argument("--optimised",
        required=False,
        default=False,
        help="enable flag for eff model",
        action='store_true')
    parser.add_argument("--alpha",
        required=False,
        help="alpha for the model",
        default=(11 / 6),
        type=float)
    parser.add_argument("--phi",
        required=False,
        help="Phi for the model.",
        default=1.0,
        type=float)
    parser.add_argument("--beta",
        required=False,
        help="Beta for the model.",
        default=None,
        type=float)
    custom_args = parser.parse_args()
    main(custom_args)
