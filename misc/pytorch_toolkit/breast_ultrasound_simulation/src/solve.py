import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import os
import shutil
import numpy as np
import time
from model import GeneratorModel, DiscriminatorModel, GeneratorInter, Generator3dInter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())


class solver():
    def __init__(self, args, train_data=None, test_data=None, restore=0):
        self.args = args
        self.train_data = train_data
        self.test_data = test_data

        self.gen = GeneratorModel(1).to(device)
        self.dis = DiscriminatorModel(2).to(device)

        self.recon_l = nn.L1Loss().to(device)
        self.adv_l = nn.BCELoss().to(device)

        self.g_opt = optim.Adam(self.gen.parameters(), lr=args.lr)
        self.d_opt = optim.Adam(self.dis.parameters(), lr=args.lr)

        self.output_dir = os.path.join('../infer_results', args.name)
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

        self.log_dir = '../logs/' + args.name
        self.writer = SummaryWriter(self.log_dir)
        if not os.path.isdir("../checkpoints/" + args.name):
            os.mkdir("../checkpoints/" + args.name)

        self.log = args.log_step != -1
        print("Log Dir: ../logs")
        print("Checkpoint Dir: ../checkpoints")

        if restore or args.test:
            state_dict = torch.load(
                os.path.join(
                    "../checkpoints",
                    args.model_name))
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

        self.g_opt.zero_grad()
        self.reconstruction_loss = self.recon_l(
            self.real, self.fake) * self.args.beta_reco
        self.generator_adversarial_loss = self.adv_l(
            self.discriminator_output, 1 - self.y) * self.args.beta_adv
        self.total_gloss = self.reconstruction_loss + self.generator_adversarial_loss
        self.total_gloss.backward()
        self.g_opt.step()

    def d_back(self):
        self.d_opt.zero_grad()

        discriminator_output = self.dis(self.discriminator_input.detach())
        self.adversarial_loss = self.adv_l(discriminator_output, self.y)

        self.d_opt.zero_grad()
        self.adversarial_loss.backward()
        self.d_opt.step()

    def train(self):

        args = self.args
        for epoch in range(args.epochs):
            L1_epoch = 0
            AdvG_epoch = 0
            AdvD_epoch = 0
            DisAcc_epoch = 0

            print(
                "\nEpoch = {}  Lr = {}".format(
                    epoch,
                    self.g_opt.state_dict()["param_groups"][0]["lr"]))
            start_time = time.time()
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
                temp = (
                    self.discriminator_output.squeeze() > 0.5).type(
                    torch.cuda.FloatTensor)
                DisAcc_epoch += (self.y ==
                                 temp).type(torch.cuda.FloatTensor).mean().item()

                # Cumulative loss for analysis
                L1_epoch += self.reconstruction_loss.item()
                AdvG_epoch += self.generator_adversarial_loss.item()
                AdvD_epoch += self.adversarial_loss.item()
                # print(time.time())
                # Python Logging
                print('\rStep [{}/{}], L1: {:.4f}, AccD: {:.4f}, AdvD: {:.4f}, AdvG: {:.4f} Time: {:.1f}sec'.format((i_batch + 1),
                                                                                                                    len(self.train_data),
                                                                                                                    L1_epoch / (i_batch + 1),
                                                                                                                    DisAcc_epoch / (i_batch + 1),
                                                                                                                    AdvD_epoch / (i_batch + 1),
                                                                                                                    AdvG_epoch / (i_batch + 1),
                                                                                                                    time.time() - start_time),
                      end=" ")

                # tensorboard logging
                if self.log:
                    if i_batch % args.log_step == 0:
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

                    if i_batch == 0 and epoch % self.args.vis_step == 0:
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
            if epoch % 2 == 0:
                torch.save(state_dict, os.path.join("../checkpoints/",
                           args.name + "/" + str(epoch % 20) + ".pt"))
            torch.save(
                state_dict,
                os.path.join(
                    "../checkpoints/",
                    args.name +
                    "/" +
                    "latest" +
                    ".pt"))

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


class solver_inter2d(solver):
    def __init__(self):
        super().__init__(args, train_data=None, test_data=None, restore=1)

        if restore or args.test:
            if args.model_name:
                state_dict = torch.load(
                    os.path.join(
                        "../checkpoints",
                        args.model_name))
                self.gen_old.load_state_dict(state_dict["generator1_weights"])
                
            self.gen = GeneratorInter(
                1, self.gen_old.cpu(), a=args.dilation_factor)
            if args.model_name:
                self.gen.load_state_dict(
                    state_dict["generator1_weights"], strict=False)

            self.gen.to(device)

            print("Loaded Model for inferencing IVUS2D")


class solver_inter3d(solver):
    def __init__(self):
        super().__init__(args, train_data=None, test_data=None, restore=1)

        if restore or args.test:
            if args.model_name:
                state_dict = torch.load(
                    os.path.join(
                        "../checkpoints",
                        args.model_name))
                self.gen_old.load_state_dict(state_dict["generator1_weights"])
                
            self.gen = Generator3dInter(1)
            self.gen2d = GeneratorInter(
                1, self.gen_old.cpu(), a=args.dilation_factor)
            if args.model_name:
                self.gen2d.load_state_dict(
                    state_dict["generator1_weights"], strict=False)

            state_dict_2d = self.gen2d.state_dict()
            state_dict_3d = self.gen.state_dict()
            keys_2d = list(self.gen2d.state_dict().keys())
            keys_3d = list(self.gen.state_dict().keys())

            c = 0
            for key in keys_3d:
                if key in keys_2d:
                    wt2D = state_dict_2d[key]
                    if len(wt2D.size()) > 1:
                        state_dict_3d[key] = torch.unsqueeze(wt2D, 2)
                        c += 1
                    else:
                        state_dict_3d[key] = wt2D
                        c += 1
                elif key.split(".")[-2] == 'ydim':
                    key_w = key.replace("ydim.weight", "wts")
                    wt2D = state_dict_2d[key_w]
                    if len(wt2D.size()) > 1:
                        state_dict_3d[key] = torch.unsqueeze(
                            wt2D, 2).permute(0, 1, 4, 3, 2)
                        c += 1
                    else:
                        state_dict_3d[key] = wt2D
                        c += 1
                else:
                    # the zdim
                    key_z = key.replace("zdim", "xdim")
                    if key_z in keys_3d:
                        wt2D = state_dict_2d[key_z]
                    if len(wt2D.size()) > 1:
                        state_dict_3d[key] = torch.unsqueeze(
                            wt2D, 2).permute(0, 1, 4, 3, 2)
                        c += 1
                    else:
                        state_dict_3d[key] = wt2D
                        c += 1
            self.gen.load_state_dict(state_dict_3d)
            self.gen.to(device)

            print("Loaded Model for inferencing IVUS3D")

    def test(self):
        self.gen.eval()
        print("Starting Testing")

        for i_batch, sample in enumerate(self.test_data):
            X, n = sample
            X = X.to(device)

            # generating fake images
            with torch.no_grad():
                self.fake = self.gen(X)

            for i in range(X.shape[0]):
                np.save(os.path.join(self.output_dir, str(i)),
                        self.fake[i].detach().cpu().numpy())

            print('Step:{}'.format(i_batch))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

