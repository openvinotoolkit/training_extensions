import os
import torch
import torchvision
from torch.backends import cudnn
import numpy as np
from termcolor import colored
import json
import random
from torch.utils import data
from dataloader import CustomDatasetPhase1, CustomDatasetPhase2
from model import AutoEncoder, Decoder
from evaluators import compare_psnr_batch, compare_ssim_batch
cudnn.benchmark = True


def get_efficient_net_parameters(iterate, phi):
    b = [i/100. for i in range(10, int(phi*100))]
    alpha = random.sample(b, iterate)
    alpha = np.array(alpha)
    beta = [np.sqrt(phi/i) for i in np.array(alpha)]
    beta = np.array(beta)
    return alpha, beta


def load_model(eff_flag=False, it_no=0, alpha=1, beta=1, depth=3, width=64, phase=1):
    if phase == 1:
        if eff_flag:
            model = AutoEncoder(
                (alpha[it_no]*depth).astype(int), (beta[it_no]*width).astype(int))
        else:
            model = AutoEncoder(int(alpha*depth), int(beta*width))
    else:
        if eff_flag:
            model = Decoder((alpha[it_no]*depth).astype(int),
                            (beta[it_no]*width).astype(int))
        else:
            model = Decoder(int(alpha*depth), int(beta*width))

    return model


def my_collate(batch):
    final_data_256 = []
    final_target_256 = []
    final_data_128 = []
    final_target_128 = []
    data = [torch.Tensor(item[0]) for item in batch]
    target = [item[1] for item in batch]
    for i in range(len(target)):
        if target[i].shape[-1] == 256:
            final_data_256.append(data[i])
            final_target_256.append(target[i])
        else:
            final_data_128.append(data[i])
            final_target_128.append(target[i])

    yield torch.stack(final_data_256), torch.stack(final_target_256)
    yield torch.stack(final_data_128), torch.stack(final_target_128)


def train_model_phase1(config, train_dataset, model,
                       optimizer, msecrit, epoch,
                       alpha, beta, it_no):

    for idx, (images, labels) in enumerate(train_dataset):
        # The data fetch loop
        if torch.cuda.is_available() and config['gpu']:
            images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()  # zero out grads

        output = model(images)
        loss = msecrit(output, labels)
        # compute the required metrics
        ssim = compare_ssim_batch(
            labels.detach().cpu().numpy(), output.detach().cpu().numpy())
        psnr = compare_psnr_batch(
            labels.detach().cpu().numpy(), output.detach().cpu().numpy())
        psnr = 20.0 * np.log10(psnr)

        if config['efficient_net']:
            if idx % config['interval'] == 0:
                print('{tag} {0:4d}/{1:4d}/{2:4d} -> Loss: {3:.8f}, \
                        pSNR: {4:.8f}dB, SSIM: {5:.8f}, alpha: {6: .8f}, \
                        beta: {7: .8f}'.format(idx, epoch, config['epochs'],
                                               loss.item(
                ), psnr, ssim, alpha[it_no], beta[it_no],
                    tag=colored('[Training]', 'yellow')))
        else:
            if idx % config['interval'] == 0:
                print('{tag} {0:4d}/{1:4d}/{2:4d} -> Loss: {3:.8f}, \
                        pSNR: {4:.8f}dB, \
                        SSIM: {5:.8f}'.format(idx, epoch, config['epochs'],
                                              loss.item(), psnr, ssim,
                                              tag=colored('[Training]', 'yellow')))

        loss.backward()  # backward
        optimizer.step()  # weight update


def train_model_phase2(config, train_dataloader, model,
                       optimizer, msecrit, epoch,
                       alpha, beta, i):

    for idx, (imageset1, imageset2) in enumerate(train_dataloader):
        loss1, loss2 = 0., 0.
        psnr1, psnr2 = 0., 0.
        ssim1, ssim2 = 0., 0.

        for i in range(len(imageset1[0])):
            data11 = imageset1[0][i]
            data12 = imageset1[1][i]
            images = data11
            labels = torch.reshape(data12, (1, 1, 256, 256))
            if torch.cuda.is_available() and config['gpu']:
                images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()  # zero out grads
            output = model(images)
            loss1 += msecrit(output, labels)
            # compute the required metrics
            ssim1 += compare_ssim_batch((labels).detach().cpu().numpy(),
                                        (output).detach().cpu().numpy())
            psnr1 += compare_psnr_batch((labels).detach().cpu().numpy(),
                                        (output).detach().cpu().numpy())

        for j in range(len(imageset2[0])):
            data21 = imageset2[0][j]
            data22 = imageset2[1][j]
            images = data21
            labels = torch.reshape(data22, (1, 1, 128, 128))

            if torch.cuda.is_available() and config['gpu']:
                images, labels = images.cuda(), labels.cuda()
            output = model(images)
            loss2 += msecrit(output, labels)

            # compute the required metrics
            ssim2 += compare_ssim_batch(labels.detach().cpu().numpy(),
                                        output.detach().cpu().numpy())
            psnr2 += compare_psnr_batch(labels.detach().cpu().numpy(),
                                        output.detach().cpu().numpy())

        loss = (loss1/(len(imageset1[0])) + loss2/(len(imageset2[0])))/2
        ssim = (ssim1/(len(imageset1[0])) + ssim2/(len(imageset2[0])))/2
        psnr = (psnr1/(len(imageset1[0])) + psnr2/(len(imageset2[0])))/2
        psnr = 20.0 * np.log10(psnr)

        if config['efficient_net']:
            if idx % config['interval'] == 0:
                print('{tag} {0:4d}/{1:4d}/{2:4d} -> Loss: {3:.8f}, \
                pSNR: {4:.8f}dB, SSIM: {5:.8f}, alpha: {6: .8f}, \
                beta: {7: .8f}'.format(idx, epoch, config['epochs'],
                                       loss.item(
                ), psnr, ssim, alpha[i], beta[i],
                    tag=colored('[Training]', 'yellow')))
        else:
            if idx % config['interval'] == 0:
                print('{tag} {0:4d}/{1:4d}/{2:4d} -> Loss: {3:.8f}, \
                pSNR: {4:.8f}dB, SSIM: {5:.8f}'.format(idx, epoch,
                                                       config['epochs'], loss.item(
                                                       ), psnr, ssim,
                                                       tag=colored('[Training]', 'yellow')))

        loss.backward()  # backward
        optimizer.step()  # weight update


def validate_model_phase1(config, test_dataloader, model, msecrit):
    for idx, (images, labels) in enumerate(test_dataloader):
        if torch.cuda.is_available() and config['gpu']:
            images, labels = images.cuda(), labels.cuda()

        output = model(images)
        loss = msecrit(output, labels)  # loss calculation
        # calculate the metrics (SSIM and pSNR)
        ssim = compare_ssim_batch(
            labels.detach().cpu().numpy(), output.detach().cpu().numpy())
        psnr = compare_psnr_batch(
            labels.detach().cpu().numpy(), output.detach().cpu().numpy())

        avg_loss = ((n * avg_loss) + loss.item()) / (n + 1)  # running mean
        avg_ssim = ((n * avg_ssim) + ssim) / (n + 1)  # running mean
        avg_psnr = ((n * avg_psnr) + psnr) / (n + 1)  # running mean
        n += 1
    return avg_loss, avg_ssim, avg_psnr


def validate_model_phase2(config, test_dataloader, model, msecrit):
    for idx, (images, labels) in enumerate(test_dataloader):
        if torch.cuda.is_available() and config['gpu']:
            images, labels = images.cuda(), labels.cuda()

        output = model(images)
        loss = msecrit(output, labels)  # loss calculation
        # calculate the metrics (SSIM and pSNR)
        ssim = compare_ssim_batch(
            labels.detach().cpu().numpy(), output.detach().cpu().numpy())
        psnr = compare_psnr_batch(
            labels.detach().cpu().numpy(), output.detach().cpu().numpy())

        avg_loss = ((n * avg_loss) + loss.item()) / (n + 1)  # running mean
        avg_ssim = ((n * avg_ssim) + ssim) / (n + 1)  # running mean
        avg_psnr = ((n * avg_psnr) + psnr) / (n + 1)  # running mean
        n += 1
    return avg_loss, avg_ssim, avg_psnr


def train_model(config):
    # image/mask transformations (just grayscaling and PIL-to-Tensor conversion)
    images_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.Grayscale(), torchvision.transforms.ToTensor()])
    labels_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.Grayscale(), torchvision.transforms.ToTensor()])

    if config['phase'] == 1:
        # Dataset & dataloader for training
        train_dataset = CustomDatasetPhase1(config['traindata'],
                                            transform_images=images_transforms,
                                            transform_masks=labels_transforms)
        train_dataset.choose_random_subset(config['subset_size'])
        train_dataloader = data.DataLoader(train_dataset,
                                           batch_size=config['batch_size'],
                                           num_workers=16, pin_memory=True,
                                           shuffle=True)

        # CBISDDSM dataset & dataloader for inference
        test_dataset = CustomDatasetPhase1(config['testdata'],
                                           transform_images=images_transforms,
                                           transform_masks=labels_transforms)
        test_dataloader = data.DataLoader(test_dataset,
                                          batch_size=1, num_workers=16,
                                          pin_memory=True, shuffle=False)

    else:
        path_train_latent = config['traindata'] + "/latent/d_1/"
        path_train_gdtruth = config['traindata'] + "/gd_truth/"

        train_dataset = CustomDatasetPhase2(path_train_latent,
                                            path_train_gdtruth, transform_images=None,
                                            transform_masks=labels_transforms, mod=0)
        train_dataloader = data.DataLoader(train_dataset,
                                           batch_size=config['batch_size'],
                                           num_workers=0, pin_memory=True,
                                           shuffle=True, collate_fn=my_collate)

        # Dataset & dataloader for inference
        path_test_latent = config['testdata'] + "/latent/d_1/"
        path_test_gdtruth = config['testdata'] + "/gd_truth/"

        test_dataset = CustomDatasetPhase2(path_test_latent,
                                           path_test_gdtruth, transform_images=None,
                                           transform_masks=labels_transforms, mod=1)
        test_dataloader = data.DataLoader(test_dataset,
                                          batch_size=1, num_workers=0,
                                          pin_memory=True, shuffle=False)

    if config['efficient_net']:
        phi = config['phi']
        iterate = config['iterate']
        alpha, beta = get_efficient_net_parameters(iterate, phi)

    else:
        if config['phase'] == 1:
            iterate, alpha, beta = 1, 1, 1
        else:
            iterate, alpha, beta = 1, 0.65, 1.5

    for i in range(iterate):
        model = load_model(eff_flag=config['efficient_net'],
                           it_no=i, alpha=alpha, beta=beta,
                           depth=config['depth'], width=config['width'],
                           phase=config['phase'])
        if torch.cuda.is_available() and config['gpu']:
            model = model.cuda()
        # usual optimizer instance
        optimizer = torch.optim.Adam(model.parameters())
        schedular = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.75)  # schedular instance
        start_epoch = 0
        model_file = '.'.join([config['model_file_name'], 'model'])

        if iterate == 1:
            start_epoch = 0
            # load model file (if present) and load the model,optimizer & schedular states
            model_file = '.'.join([config['model_file_name'], 'model'])
            if os.path.exists(os.path.abspath(model_file)):
                loaded_file = torch.load(os.path.abspath(model_file))
                model.load_state_dict(loaded_file['model_state'])
                optimizer.load_state_dict(loaded_file['optim_state'])
                schedular.load_state_dict(loaded_file['schedular_state'])
                start_epoch = loaded_file['epoch'] + 1

                print('{tag} resuming from saved model'.format(
                    tag=colored('[Saving]', 'red')))
                del loaded_file

        # For logging purpose; read anch chk for val decr : val in 2nd row < 1st row
        logg = []
        log_file = '.'.join([config['model_file_name'], 'log'])
        if os.path.exists(os.path.abspath(log_file)):
            # if .log file exists, open it
            with open(log_file, 'r') as logfile:
                logg = json.load(logfile)

        msecrit = torch.nn.MSELoss()  # usual MSE loss function
        prev_test_ssim = -np.inf

        for epoch in range(start_epoch, config['epochs']):
            schedular.step()
            # choose randomize subset if requested
            if config['randomize_subset']:
                train_dataset.choose_random_subset(config['subset_size'])

            model.train()  # training mode ON

            if config['phase'] == 1:
                train_model_phase1(config, train_dataloader, model,
                                   optimizer, msecrit, epoch,
                                   alpha, beta, i)
            else:
                train_model_phase2(config, train_dataloader, model,
                                   optimizer, msecrit, epoch,
                                   alpha, beta, i)

            # TRAINING DONE
            model.eval()  # switch to evaluation mode

            n = 0
            avg_loss, avg_ssim, avg_psnr = 0.0, 0.0, 0.0
            with torch.no_grad():
                # Testing phase starts
                if config['phase'] == 1:
                    validate_model_phase1(
                        config, test_dataloader, model, msecrit)
                else:
                    validate_model_phase2(
                        config, test_dataloader, model, msecrit)  # later
                    pass

            avg_psnr = 20.0 * np.log10(avg_psnr)  # convert pSNR to dB

            print('{tag} Epoch: {0:3d}, Loss: {3:.8f}, pSNR: {1:.8f}, SSIM: {2:.8f}'.format(
                epoch, avg_psnr, avg_ssim, avg_loss, tag=colored('[Testing]', 'cyan')))
            if config['efficient_net']:
                aa = alpha[i]
                bb = beta[i]
            else:
                aa = alpha
                bb = beta
            logg.append(
                {
                    'epoch': epoch,
                    'loss': avg_loss,
                    'pSNR': avg_psnr,
                    'SSIM': avg_ssim,
                    'lr': optimizer.param_groups[0]['lr'],
                    'Alpha': aa,
                    'Beta': bb
                })  # accumulate information for the .log file

            # name of the log file
            with open(log_file, 'w') as logfile:
                json.dump(logg, logfile)

            # model saving, only if the SSIM is better than before
            if avg_ssim > prev_test_ssim:
                print(
                    colored('[Saving] model saved to {}'.format(model_file), 'red'))
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optim_state': optimizer.state_dict(),
                    'schedular_state': schedular.state_dict(),
                }, os.path.abspath(model_file))
                prev_test_ssim = avg_ssim
            else:
                print(colored('[Saving] model NOT saved'))
