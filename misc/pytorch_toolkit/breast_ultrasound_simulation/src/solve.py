import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import os
import numpy as np
import time
from .model import GeneratorModel, DiscriminatorModel, GeneratorInter
import random
from ax.service.ax_client import AxClient
import onnxruntime
from torchvision import transforms
from openvino.inference_engine import IECore

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.use_deterministic_algorithms(True)


class solver():
    def __init__(self, train_config, gen_config, train_data=None, test_data=None, restore=0):
        self.train_config = train_config
        self.gen_config = gen_config
        self.train_data = train_data
        self.test_data = test_data

        self.gen = GeneratorModel(1).to(device)
        self.dis = DiscriminatorModel(2).to(device)

        self.recon_l = nn.L1Loss().to(device)
        self.adv_l = nn.BCELoss().to(device)

        self.g_opt = optim.Adam(self.gen.parameters(), lr=self.train_config['lr'])
        self.d_opt = optim.Adam(self.dis.parameters(), lr=self.train_config['lr'])

        self.output_dir = os.path.join('temp_data','infer_results', self.train_config['model_name'])
        if(not os.path.isdir(self.output_dir)):
            os.mkdir(self.output_dir)

        log_path = os.path.join('temp_data','logs', self.train_config['model_name'])

        self.log_dir = log_path
        self.writer = SummaryWriter(self.log_dir)
        chck_path = os.path.join("temp_data","checkpoints", self.train_config['model_name'])
        if(not os.path.isdir(chck_path)):
            os.mkdir(chck_path)

        self.log = self.train_config['log_step'] != -1
        print(f"Log Dir: {log_path}")
        print(f"Checkpoint Dir: {chck_path}")

        if restore or gen_config['test_flag']:
            state_dict = torch.load(
                os.path.join(
                    "downloads","checkpoints",
                    self.train_config['model_weight']))
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
            self.real, self.fake) * self.train_config['beta_recovery']
        self.generator_adversarial_loss = self.adv_l(
            self.discriminator_output, 1 - self.y) * self.train_config['beta_adv']
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

        self.epochs = num_epochs_to_run

        if best_param is None:
            if model_gen is not None:
                self.gen = model_gen
            if optim_gen is not None:
                self.g_opt = optim_gen
        else:
            self.gen = GeneratorModel(1).to(device)
            self.g_opt = optim.Adam(
                self.gen.parameters(),
                lr=best_param.get('lr1'),
                weight_decay=best_param.get('weight_decay1'))

        for epoch in range(self.epochs):

            L1_epoch = 0
            AdvG_epoch = 0
            AdvD_epoch = 0
            DisAcc_epoch = 0

            print(f"Epoch = {epoch}  Lr = {self.g_opt.state_dict()['param_groups'][0]['lr']}")
            start_time = time.time()
            for i_batch, sample in enumerate(self.train_data):
                X, real, _ = sample
                X = X.to(device)
                self.real = real.to(device)

                # generating fake images
                self.fake = self.gen(X)

                # channel shuffler with labels
                self.discriminator_input, self.y = self.shuffler(self.real, self.fake)

                self.set_requires_grad(self.dis, False)
                self.g_back()
                self.set_requires_grad(self.dis, True)

                self.set_requires_grad(self.gen, False)
                self.d_back()
                self.set_requires_grad(self.gen, True)

                # Accuracy of disc
                temp = (self.discriminator_output.squeeze() > 0.5).type(torch.cuda.FloatTensor)
                DisAcc_epoch += (self.y == temp).type(torch.cuda.FloatTensor).mean().item()

                # Cumulative loss for analysis
                L1_epoch += self.reconstruction_loss.item()
                AdvG_epoch += self.generator_adversarial_loss.item()
                AdvD_epoch += self.adversarial_loss.item()
                print(f'Step [{i_batch + 1}/{len(self.train_data)}], L1: {L1_epoch / (i_batch + 1)}')
                print(f' AccD: {DisAcc_epoch / (i_batch + 1)}, AdvD: {AdvD_epoch / (i_batch + 1)}')
                print(f'AdvG: {AdvG_epoch / (i_batch + 1)}')

                # tensorboard logging
                if(self.log):
                    if(i_batch % self.train_config['log_step'] == 0):
                        self.writer.add_scalar('Loss/Discriminator',
                                               self.adversarial_loss.item(),
                                               i_batch + len(self.train_data) * epoch)
                        self.writer.add_scalar('Loss/Generator',
                                               self.generator_adversarial_loss.item(),
                                               i_batch + len(self.train_data) * epoch)
                        self.writer.add_scalar('Loss/Reconstruction',
                                               self.reconstruction_loss.item(),
                                               i_batch + len(self.train_data) * epoch)
                        self.writer.add_scalar('LR', get_lr(
                            self.g_opt), i_batch + len(self.train_data) * epoch)

                    if(i_batch == 0 and epoch % self.train_config['vis_step'] == 0):
                        self.writer.add_image('Fake', torchvision.utils.make_grid(self.fake[:16].detach(
                        ).cpu(), scale_each=True), i_batch + len(self.train_data) * epoch)
                        self.writer.add_image('Real', torchvision.utils.make_grid(self.real[:16].detach(
                        ).cpu(), scale_each=True), i_batch + len(self.train_data) * epoch)
                        self.writer.add_image('stage0', torchvision.utils.make_grid(
                            X[:16].detach().cpu(), scale_each=True), i_batch + len(self.train_data) * epoch)
                    self.writer.flush()

            state_dict = {
                "generator1_weights": self.gen.state_dict(),
                "discriminator1_weights": self.dis.state_dict(),
                "epoch": epoch,
                "generator1_optimizer": self.g_opt.state_dict(),
                "discriminator1_optimizer": self.d_opt.state_dict()}
            if(epoch % 2 == 0):
                savepath = os.path.join("temp_data", "checkpoints", self.gen_config['exp_name'])
                if not os.path.exists(savepath):
                    os.mkdir(savepath)
                torch.save(state_dict, os.path.join(savepath, str(epoch % 20)+".pt") )
            torch.save(state_dict, os.path.join("temp_data", "checkpoints", self.gen_config['exp_name'], "latest.pt"))

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
                torchvision.utils.save_image(
                    self.fake[i].detach().cpu(), os.path.join(
                        self.output_dir, n[i]), normalize=True, scale_each=True)

            print('Step:{}'.format(i_batch))

    def test_optimizer(self):
        self.gen.eval()
        self.dis.eval()
        self.total_gloss = 0.0

        print("Starting Testing")
        for i_batch, sample in enumerate(self.test_data):
            X, y, n = sample
            X = X.to(device)
            self.real = y.to(device)

            with torch.no_grad():
                self.fake = self.gen(X)

                self.discriminator_input, self.y = self.shuffler(self.real, self.fake)
                self.discriminator_output = self.dis(self.discriminator_input)
                self.discriminator_output = torch.flatten(self.discriminator_output)

                self.reconstruction_loss = self.recon_l(self.real, self.fake) * self.args.beta_reco
                self.generator_adversarial_loss = self.adv_l(self.discriminator_output, 1 - self.y) * self.args.beta_adv
                self.total_gloss = self.total_gloss + self.reconstruction_loss.item() + self.generator_adversarial_loss.item()
        self.gen.train()
        self.dis.train()
        return (1.0 / self.total_gloss)

    def evaluate_bus(self, parameters):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        model_gen = GeneratorModel(1).to(device)
        optim_gen = optim.Adam(model_gen.parameters(), lr=parameters.get("lr1", 0.001), weight_decay=parameters.get("weight_decay1", 0.0))

        num_epochs_to_run = 20
        '''for epoch in range(5):  # 50
            self.train(5)'''
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

        for _ in range(40):  # 50
            parameters, trial_index = ax_client.get_next_trial()
            ax_client.complete_trial(trial_index=trial_index, raw_data=self.evaluate_bus(parameters))

        best_parameters, metrics = ax_client.get_best_parameters()
        print(best_parameters)
        return best_parameters

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def load_inference_model(run_type, onnx_checkpoint, infer_config, restore, test_flag):

    if run_type == 'pytorch':
        if restore or test_flag:
            if(infer_config['model_name']):
                gen_old = GeneratorModel(1).to(device)
                state_dict_path = os.path.join("temp_data","checkpoints", infer_config['exp_name'],"latest.pt")
                state_dict = torch.load(state_dict_path)
                gen_old.load_state_dict(state_dict["generator1_weights"])
                
            gen_new = GeneratorInter(1, gen_old.cpu(), a=infer_config['dilation_factor'])
            if(infer_config['model_name']):
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

class solver_inter2d(solver):
    def __init__(self, infer_config, test_flag, train_data=None, test_data=None, restore=1, run_type='pytorch'):
        self.infer_config = infer_config
        self.train_data = train_data
        self.test_data = test_data
        self.output_dir = os.path.join('temp_data','infer_results', infer_config["exp_name"])
        self.gen_old = GeneratorModel(1).to(device)
        self.restore = restore
        self.test_flag = test_flag
        self.run_type = run_type
        self.onnx_checkpoint = self.infer_config['onnx_checkpoint']
        if(not os.path.isdir(self.output_dir)):
            os.makedirs(self.output_dir)

    def test(self):
        print("Starting Testing")
        model = load_inference_model(self.run_type,
                                    self.onnx_checkpoint,
                                    self.infer_config,
                                    self.restore,
                                    self.test_flag)

        for i_batch, sample in enumerate(self.test_data):
            X, _,_ = sample
            X = X.to(device)
            to_tensor = transforms.ToTensor()
            # generating fake images
            with torch.no_grad():
                if self.run_type == 'pytorch':
                    out = model(X)
                    self.fake = out
                elif self.run_type == 'onnx':
                    ort_inputs = {model.get_inputs()[0].name: to_numpy(X)}
                    out = model.run(None, ort_inputs)
                    out = np.array(out)
                    out = out.squeeze(0)
                    out = to_tensor(out.squeeze(0)).to(device)
                    self.fake = out
                else:
                    out = model.infer(inputs={'input': to_numpy(X)})['output']
                    out = out.squeeze(0)
                    out = to_tensor(out).squeeze(0)
                    self.fake = out

            for i in range(X.shape[0]):
                np.save(os.path.join(self.output_dir, str(i)), self.fake[i].detach().cpu().numpy())

            print('Step:{}'.format(i_batch))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
