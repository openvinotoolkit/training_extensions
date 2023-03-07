import torch
from torch import nn
from torch import optim
import torchvision
import os
import numpy as np
import time
from breast_ultrasound_simulation.stage1.src.model import (GeneratorModel,
                                                           DiscriminatorModel,
                                                           GeneratorInter)
import random
from ax.service.ax_client import AxClient
import onnxruntime
from openvino.inference_engine import IECore
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.use_deterministic_algorithms(True)


class solver():
    def __init__(self, args, train_data=None, test_data=None, restore=0):
        self.args = args
        self.train_data = train_data
        self.test_data = test_data

        self.gen = GeneratorModel(1).to(device)
        self.dis = DiscriminatorModel(2).to(device)
        self.gen_start = GeneratorModel(1).to(device)
        self.dis_start = DiscriminatorModel(2).to(device)
        self.gen_start.load_state_dict(self.gen.state_dict())
        self.dis_start.load_state_dict(self.dis.state_dict())

        self.recon_l = nn.L1Loss().to(device)
        self.adv_l = nn.BCELoss().to(device)

        self.g_opt = optim.Adam(self.gen.parameters(), lr=args['lr'])
        self.d_opt = optim.Adam(self.dis.parameters(), lr=args['lr'])

        if args['openvino'] == 1:
            self.output_dir = os.path.join('downloads/test_outputs/outputs')
            if not os.path.isdir(self.output_dir):
                os.makedirs(self.output_dir)

            if not os.path.isdir('downloads/checkpoints_train'):
                os.makedirs('downloads/checkpoints_train')

        else:
            self.output_dir = os.path.join('breast_ultrasound_simulation/stage1/infer_results', args['name'])
            if not os.path.isdir(self.output_dir):
                os.makedirs(self.output_dir)

            if not os.path.isdir("breast_ultrasound_simulation/stage1/checkpoints/" + args['name']):
                os.makedirs("breast_ultrasound_simulation/stage1/checkpoints/" + args['name'])

        self.log = args['log_step'] != -1

        if restore or args['test']:
            state_dict = torch.load(
                os.path.join(
                    "breast_ultrasound_simulation/stage1/checkpoints",
                    args['model_name']))
            self.gen.load_state_dict(state_dict["generator1_weights"])
            self.dis.load_state_dict(state_dict["discriminator1_weights"])
            self.g_opt.load_state_dict(state_dict["generator1_optimizer"])
            self.d_opt.load_state_dict(state_dict["discriminator1_optimizer"])
            print("restored")

    def set_requires_grad(self, net, requires_grad=False):
        """Set :requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not"""

        for param in net.parameters():
            param.requires_grad = requires_grad

    def shuffler(self, real, fake):
        y = torch.randint(0, 2, (real.size(0),), dtype=torch.float).to(device)

        z1 = torch.cat([real, fake], dim=1)
        z2 = torch.cat([fake, real], dim=1)

        discriminator_input = torch.empty(
            (0, z1.size(1), z1.size(2), z1.size(3))).to(device)

        for i in range(y.size(0)):
            if y[i].item() == 1:
                discriminator_input = torch.cat(
                    [discriminator_input, z1[i:i + 1]], dim=0)
            if y[i].item() == 0:
                discriminator_input = torch.cat(
                    [discriminator_input, z2[i:i + 1]], dim=0)

        return discriminator_input, y

    def g_back(self):

        self.discriminator_output = self.dis(self.discriminator_input)
        self.discriminator_output = torch.flatten(self.discriminator_output)

        self.g_opt.zero_grad()
        self.reconstruction_loss = self.recon_l(
            self.real, self.fake) * self.args['beta_reco']
        self.generator_adversarial_loss = self.adv_l(
            self.discriminator_output, 1 - self.y) * self.args['beta_adv']
        self.total_gloss = self.reconstruction_loss + self.generator_adversarial_loss
        self.total_gloss.backward()
        self.g_opt.step()

    def d_back(self):
        self.d_opt.zero_grad()

        discriminator_output = self.dis(self.discriminator_input.detach())
        discriminator_output = torch.flatten(discriminator_output)
        self.adversarial_loss = self.adv_l(discriminator_output, self.y)

        self.d_opt.zero_grad()
        self.adversarial_loss.backward()
        self.d_opt.step()

    def train(self, num_epochs_to_run=500, model_gen=None, optim_gen=None, best_param=None):

        args = self.args
        args['epochs'] = num_epochs_to_run

        if best_param is None:
            if model_gen is not None:
                self.gen = model_gen
            if optim_gen is not None:
                self.g_opt = optim_gen
        else:
            self.gen = GeneratorModel(1).to(device)
            self.g_opt = optim.Adam(self.gen.parameters(),
                                    lr=best_param.get('lr1'),
                                    weight_decay=best_param.get('weight_decay1'))

        self.gen.load_state_dict(self.gen_start.state_dict())
        self.dis.load_state_dict(self.dis_start.state_dict())

        losses_accross_epochs = []

        for epoch in range(args['epochs']):

            L1_epoch = 0
            AdvG_epoch = 0
            AdvD_epoch = 0
            DisAcc_epoch = 0

            print(f'Epoch: {epoch} | LR: {self.g_opt.state_dict()["param_groups"][0]["lr"]}')

            for i_batch, sample in enumerate(self.train_data):
                X, real, _ = sample

                X = X.to(device)

                self.real = real.to(device)

                # generating fake images
                self.fake = self.gen(X)

                # channel shuffler with labels
                self.discriminator_input, self.y = self.shuffler(
                    self.real, self.fake)

                self.set_requires_grad(self.dis, False)
                self.g_back()
                self.set_requires_grad(self.dis, True)

                self.set_requires_grad(self.gen, False)
                self.d_back()
                self.set_requires_grad(self.gen, True)

                # Accuracy of disc

                temp = (self.discriminator_output.squeeze() > 0.5).type(torch.float32)
                DisAcc_epoch += (self.y == temp).type(torch.float32).mean().item()

                # Cumulative loss for analysis
                L1_epoch += self.reconstruction_loss.item()
                AdvG_epoch += self.generator_adversarial_loss.item()
                AdvD_epoch += self.adversarial_loss.item()

                # Python Logging

                print(f'Step:[{(i_batch + 1)/len(self.train_data)}] | \
                        L1: {L1_epoch / (i_batch + 1)} \
                        AccD: {DisAcc_epoch / (i_batch + 1)} | \
                        AdvD: { AdvD_epoch / (i_batch + 1)} | \
                        AdvG: {AdvG_epoch / (i_batch + 1)}')

                losses_accross_epochs.append((L1_epoch + AdvG_epoch) / (i_batch + 1))

            state_dict = {
                "generator1_weights": self.gen.state_dict(),
                "discriminator1_weights": self.dis.state_dict(),
                "epoch": epoch,
                "generator1_optimizer": self.g_opt.state_dict(),
                "discriminator1_optimizer": self.d_opt.state_dict()}
            if args['openvino'] == 1:
                if ((epoch + 1) % 10) == 0:
                    torch.save(state_dict, os.path.join("downloads/checkpoints_train/", str(epoch).zfill(4) + ".pt"))
                torch.save(state_dict, os.path.join("downloads/checkpoints_train/", "latest" + ".pt"))
            else:
                if ((epoch + 1) % 10) == 0:
                    torch.save(state_dict, os.path.join("breast_ultrasound_simulation/stage1/checkpoints/",
                                                        args['name'] + "/" + str(epoch).zfill(4) + ".pt"))
                torch.save(state_dict, os.path.join("breast_ultrasound_simulation/stage1/checkpoints/",
                                                    args['name'] + "/" + "latest" + ".pt"))

        if args['openvino'] == 1:
            return losses_accross_epochs
        return None

    def test(self):
        self.gen.eval()
        print("Starting Testing")
        avg_time = 0
        for i_batch, sample in enumerate(self.test_data):
            X, _, n = sample
            X = X.to(device)

            # generating fake images
            with torch.no_grad():
                start_time = time.time()
                self.fake = self.gen(X)
                total_time = time.time() - start_time
            avg_time += total_time

            for i in range(X.shape[0]):
                torchvision.utils.save_image(self.fake[i].detach().cpu(),
                                             os.path.join(self.output_dir, n[i]), normalize=True, scale_each=True)

            print(f'Step:{i_batch}')

    def test_optimizer(self):
        self.gen.eval()
        self.dis.eval()
        self.total_gloss = 0.0

        print("Starting Testing")
        for _, sample in enumerate(self.test_data):
            X, y, _ = sample
            X = X.to(device)
            self.real = y.to(device)

            with torch.no_grad():
                self.fake = self.gen(X)

                self.discriminator_input, self.y = self.shuffler(self.real, self.fake)
                self.discriminator_output = self.dis(self.discriminator_input)
                self.discriminator_output = torch.flatten(self.discriminator_output)

                self.reconstruction_loss = self.recon_l(self.real, self.fake) * self.args['beta_reco']
                self.generator_adversarial_loss = self.adv_l(self.discriminator_output,
                                                             1 - self.y) * self.args['beta_adv']
                self.total_gloss = self.total_gloss + \
                    self.reconstruction_loss.item() + self.generator_adversarial_loss.item()

        self.gen.train()
        self.dis.train()
        return 1.0 / self.total_gloss

    def evaluate_bus(self, parameters):
        use_cuda = torch.cuda.is_available()
        devicee = torch.device("cuda" if use_cuda else "cpu")
        _, _ = self.train_data, self.test_data
        model_gen = GeneratorModel(1).to(devicee)
        optim_gen = optim.Adam(model_gen.parameters(),
                               lr=parameters.get("lr1", 0.001),
                               weight_decay=parameters.get("weight_decay1", 0.0))

        num_epochs_to_run = 20

        self.train(num_epochs_to_run, model_gen, optim_gen)

        acc = self.test_optimizer()
        return acc

    def optimize_bayesian(self):
        ax_client = AxClient()
        ax_client.create_experiment(name='my_bayesianopt',
                                    parameters=[{"name": "lr1",
                                                 "type": "range",
                                                 "bounds": [1e-6, 1e-1],
                                                 "log_scale": True},
                                                {"name": "weight_decay1",
                                                 "type": "range",
                                                 "bounds": [0.0, 10.0],
                                                 "log_scale": False}],
                                    objective_name='evaluate_bus',
                                    minimize=False)

        for _ in range(50):  # 50
            parameters, trial_index = ax_client.get_next_trial()
            ax_client.complete_trial(trial_index=trial_index, raw_data=self.evaluate_bus(parameters))

            self.gen.load_state_dict(self.gen_start.state_dict())
            self.dis.load_state_dict(self.dis_start.state_dict())

        best_parameters, _ = ax_client.get_best_parameters()
        print(best_parameters)
        return best_parameters

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def load_inference_model(run_type, onnx_checkpoint, infer_config, restore, test_flag):

    if run_type == 'pytorch':
        if restore or test_flag:
            if infer_config['model_name']:
                gen_old = GeneratorModel(1).to(device)
                state_dict_path = os.path.join("downloads", "checkpoints", "model.pt")
                state_dict = torch.load(state_dict_path)
                gen_old.load_state_dict(state_dict["generator1_weights"])
            gen_new = GeneratorInter(1, gen_old.cpu(), a=infer_config['dilation_factor'])
            if infer_config['model_name']:
                gen_new.load_state_dict(state_dict["generator1_weights"], strict=False)
            model = gen_new.to(device)
    elif run_type == 'onnx':
        model = onnxruntime.InferenceSession(onnx_checkpoint)
    else:
        ie = IECore()
        model_xml = os.path.splitext(onnx_checkpoint)[0] + ".xml"
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        model_temp = ie.read_network(model_xml, model_bin)
        model = ie.load_network(network=model_temp, device_name='CPU')

    return model

class solver_inter2d():
    def __init__(self, infer_config, test_flag, train_data=None, test_data=None, restore=1, run_type='pytorch'):
        self.infer_config = infer_config
        self.train_data = train_data
        self.test_data = test_data
        self.output_dir = os.path.join('downloads', 'test_outputs', 'outputs')
        self.gen_old = GeneratorModel(1).to(device)
        self.restore = restore
        self.test_flag = test_flag
        self.run_type = run_type
        self.onnx_checkpoint = self.infer_config['onnx_checkpoint']
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if self.run_type == 'pytorch':
            self.output_dir = os.path.join(self.output_dir, 'pytorch')
        elif self.run_type == 'onnx':
            self.output_dir = os.path.join(self.output_dir, 'onnx')
        else:
            self.output_dir = os.path.join(self.output_dir, 'ir')

        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

    def test(self):
        print("Starting Testing")
        model = load_inference_model(self.run_type,
                                     self.onnx_checkpoint,
                                     self.infer_config,
                                     self.restore,
                                     self.test_flag)

        for i_batch, sample in enumerate(self.test_data):
            X, _, _ = sample
            X = X.to(device)
            # generating fake images
            with torch.no_grad():
                if self.run_type == 'pytorch':
                    out = model(X)
                    out = out.squeeze(0)
                    out = out.squeeze(0)
                    out = to_numpy(out)
                elif self.run_type == 'onnx':
                    ort_inputs = {model.get_inputs()[0].name: to_numpy(X)}
                    out = model.run(None, ort_inputs)
                    out = np.array(out)
                    out = out.squeeze(0).squeeze(1)
                    out = out.squeeze(0)
                else:
                    out = model.infer(inputs={'input': to_numpy(X)})['output']
                    out = out.squeeze(0)
                    out = out.squeeze(0)


            im = Image.fromarray(out)
            im = im.convert('RGB')
            im.save(self.output_dir+f'/{i_batch}.jpg')

            print(f'Step:{i_batch}')

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
