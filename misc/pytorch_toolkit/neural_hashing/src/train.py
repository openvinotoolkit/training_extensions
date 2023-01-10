import argparse
import os
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
from torchvision import  transforms
from src.utils.network import Encoder, Classifier1, Discriminator
from src.utils.dataloader import DataloderImg
from src.utils.vectorHandle import shuffler
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def main(args):

    lr = args.lr
    batch_size = args.bs
    max_epoch = args.epochs
    zsize = args.zsize
    alpha1 = args.alpha1
    alpha2 = args.alpha2
    beta = args.beta
    gamma = args.gamma
    numclasses = args.clscount
    current_dir =  os.path.abspath(os.path.dirname(__file__))
    train_path =  os.path.join(current_dir, args.dpath, 'train')
    val_path =  os.path.join(current_dir, args.dpath, 'val')

    transform = transforms.Compose([
    transforms.Resize((28, 28), transforms.InterpolationMode("bicubic")),
    transforms.ToTensor(),
    ])
    #Data loading

    trainset = DataloderImg(train_path, transform=transform, target_transform=None)
    trainloader = torch.utils.data.DataLoader(trainset, shuffle=True,
                                            batch_size=batch_size,
                                            num_workers=4)
    valset = DataloderImg(val_path, transform=transform, target_transform=None)
    valloader = torch.utils.data.DataLoader(valset, shuffle=True,
                                            batch_size=int(batch_size/2),
                                            num_workers=4)
    print("\nDataset generated. \n\n")
    #Construct Model

    encoder = Encoder(zsize)
    classifier = Classifier1(numclasses)
    discriminator = Discriminator(zsize)

    if torch.cuda.is_available():
        encoder.cuda()
        classifier.cuda()
        discriminator.cuda()

    #Train the  Model
    _, _ = Trainer(encoder, classifier, discriminator, trainloader,
                    valloader, lr, max_epoch, batch_size, zsize, alpha1,
                    alpha2, beta, gamma)
    print("Model trained !")


def Trainer(encoder, classifier, discriminator,
            trainloader, valloader, lr, max_epoch, batch_size,
            zsize, alpha1, alpha2, beta, gamma):

    similarity_loss_dict = {}
    relational_loss_dict = {}
    classifier_loss_dict = {}
    discriminator_loss_dict = {}
    current_dir =  os.path.abspath(os.path.dirname(__file__))
    savepath = os.path.join(current_dir, 'utils','model_weights')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr, eps=0.0001, amsgrad=True)
    classifier1_optimizer = optim.Adam(classifier.parameters(), lr=lr, eps=0.0001, amsgrad=True)
    discriminator_optimizer = optim.Adam(discriminator.parameters(),
                                        lr = lr/10,eps = 0.0001,amsgrad = True)
    loss_criterion = nn.BCELoss()
    ac1_loss_criterion = nn.NLLLoss()
    discriminator_criterion = nn.BCEWithLogitsLoss()

    encoder.train()

    for epoch in range(max_epoch):
        print (f"Epoch: {epoch+1}/{max_epoch}")
        similarity_loss = 0
        similarity_loss_temp = 0
        similarity_count_temp = 0
        similarity_count_full = 0

        relational_loss = 0
        relational_loss_temp = 0
        relational_count_temp = 0
        relational_count_full = 0

        classifier_loss = 0
        classifier_loss_temp = 0
        classifier_count_temp = 0
        classifier_count_full = 0

        discriminator_loss = 0
        discriminator_loss_temp = 0
        discriminator_count_temp = 0
        discriminator_count_full = 0

        hd_t0 = 0
        hd_t1 = 0
        hd_t2 = 0
        for _, data in enumerate(trainloader, 0):
            input1, input2, labels, groundtruths1, groundtruths2 = data
            indexes0 = np.where(labels.numpy() == 0)[0].tolist()
            indexes1 = np.where(labels.numpy() == 1)[0].tolist()
            indexes2 = np.where(labels.numpy() == 2)[0].tolist()

            if not len(indexes2) == batch_size:
                input1_new = torch.from_numpy(np.delete(input1.numpy(), indexes2, 0))
                input2_new = torch.from_numpy(np.delete(input2.numpy(), indexes2, 0))
                labels_2 = 1-labels[labels != 2]
                input1_new  = Variable(input1_new).cuda()
                input2_new, labels_2 = Variable(input2_new).cuda(), Variable(labels_2).cuda()
                h1_new, _ = encoder(input1_new)
                h2_new, _ = encoder(input2_new)
            input1, input2, labels = input1.cuda(), input2.cuda(), labels.cuda()
            groundtruths1, groundtruths2 = groundtruths1.cuda(), groundtruths2.cuda()

            hash1, out1 = encoder(input1)
            hash2, _ = encoder(input2)

            # Discriminator

            if len(indexes0) > 0:
                d_h1 = hash1[indexes0]
                d_h2 = hash2[indexes0]
                d_h1, d_h2, dlabels = shuffler(d_h1, d_h2)
                dlabels = Variable(dlabels).cuda()
                d_input = torch.stack((d_h1, d_h2), 1)
                d_output = discriminator(d_input.cuda()).view(-1)
                d_loss = beta*discriminator_criterion(dlabels.float(), d_output)

                discriminator_optimizer.zero_grad()
                d_loss.backward(retain_graph=True)
                discriminator_optimizer.step()
                discriminator_count_temp += 1
                discriminator_count_full += 1

            # AUXILARY CLASSIFIER1


            pred = classifier(out1)
            ac1_loss =  ac1_loss_criterion(pred.cuda(), groundtruths1)

            classifier1_optimizer.zero_grad()

            ac1_loss.backward(retain_graph=True)

            classifier1_optimizer.step()
            classifier_count_temp += 1
            classifier_count_full += 1


            # CAUCHY LOSS 1 [t =2 vs t=1,t=0]

            torch.autograd.set_detect_anomaly(True)
            del input1
            del input2
            s = (labels != 2)
            cos = F.cosine_similarity(hash1, hash2, dim=1, eps=1e-6)
            dist = F.relu((1-cos)*zsize/2)
            hd_t0 += torch.sum(dist[indexes0]).item()/(dist[indexes0].size(0) + 0.0000001)
            hd_t1 += torch.sum(dist[indexes1]).item()/(dist[indexes1].size(0) + 0.0000001)
            hd_t2 += torch.sum(dist[indexes2]).item()/(dist[indexes2].size(0) + 0.0000001)

            cauchy_output = torch.reciprocal(dist+gamma)*gamma
            try:
                loss1 = alpha2*loss_criterion(torch.squeeze(cauchy_output), s.float())
            except RuntimeError:
                print(torch.squeeze(cauchy_output))
                print(s)
                print("s", torch.max(s.float()).item(),torch.min(s.float()).item())
                print("\nCO ", torch.max(torch.squeeze(cauchy_output)).item(),
                                        torch.min(torch.squeeze(cauchy_output)).item())
            similarity_count_temp += 1
            similarity_count_full += 1

            # CAUCHY LOSS 2 [t=1 vs t=0]

            if not len(indexes2) == batch_size:
                cos = F.cosine_similarity(h1_new, h2_new, dim=1, eps=1e-6)
                dist = F.relu((1-cos)*zsize/2)
                cauchy_output = torch.reciprocal(dist+gamma)*gamma
                try:
                    loss2 = alpha2*loss_criterion(torch.squeeze(cauchy_output), labels_2.float())
                except RuntimeError:
                    print(torch.squeeze(cauchy_output))
                    print(labels_2)
                    print("s", torch.max(labels_2.float()).item(),
                            torch.min(labels_2.float()).item())
                    print("\nCO ", torch.max(torch.squeeze(cauchy_output)).item(),
                            torch.min(torch.squeeze(cauchy_output)).item())
                relational_count_temp += 1
                relational_count_full += 1
        loss=  loss1 + loss2
        encoder_optimizer.zero_grad()
        loss.backward()
        encoder_optimizer.step()

        discriminator_loss += d_loss.item()/beta
        discriminator_loss_temp += d_loss.item()/beta

        classifier_loss += ac1_loss.item()
        classifier_loss_temp += ac1_loss.item()

        similarity_loss += loss.item()/alpha1
        similarity_loss_temp += loss.item()/alpha1

        relational_loss += loss.item()/alpha2
        relational_loss_temp += loss.item()/alpha2

        similarity_loss_dict[epoch] = similarity_loss/similarity_count_full
        relational_loss_dict[epoch] = relational_loss/relational_count_full
        classifier_loss_dict[epoch] = classifier_loss/classifier_count_full
        discriminator_loss_dict[epoch] = discriminator_loss/discriminator_count_full
        print(f"similarity loss: {similarity_loss_dict[epoch]}")
        print(f"similarity loss: {relational_loss_dict[epoch]}")
        encoder_path = os.path.join(savepath, 'encoder-%d.pkl' %(epoch +1))
        torch.save(encoder.state_dict(), encoder_path)
        print('Model Saved!')
        if (epoch + 1) % 1 == 0:
            with torch.no_grad():
                t_loss1 = 0
                t_loss2 = 0
                t_run = 0
                ac1_total = 0
                ac1_correct = 0
                d_total = 0
                d_correct = 0
                print('\n Testing ....')
                for _, t_data in enumerate(valloader):

                    t_input1, t_input2, t_labels, t_gt1, t_gt2 = t_data
                    t_indexes = np.where(t_labels.numpy() == 0)[0].tolist()
                    t_indexes2 = np.where(t_labels.numpy() == 2)[0].tolist()

                    if not len(t_indexes2) == int(batch_size/2):
                        t_input1_new = torch.from_numpy(np.delete(t_input1.numpy(), t_indexes2,0))
                        t_input2_new = torch.from_numpy(np.delete(t_input2.numpy(), t_indexes2,0))
                        t_labels_2 = 1-t_labels[t_labels != 2]
                        t_input1_new = Variable(t_input1_new).cuda()
                        t_input2_new = Variable(t_input2_new).cuda()
                        t_labels_2 = Variable(t_labels_2).cuda()
                        h1_t_new, _ = encoder(t_input1_new)
                        h2_t_new, _ = encoder(t_input2_new)

                    t_input1 = Variable(t_input1).cuda()
                    t_input2 = Variable(t_input2).cuda()
                    t_labels = Variable(t_labels).cuda()
                    t_gt1, t_gt2 = Variable(t_gt1).cuda(), Variable(t_gt2).cuda()

                    s_t = (t_labels != 2)

                    h1_t, x1_t = encoder(t_input1)
                    h2_t, _ = encoder(t_input2)
                    # disc accuracy
                    if len(t_indexes) > 0:
                        d_h1_t = h1_t[t_indexes]
                        d_h2_t = h2_t[t_indexes]
                        d_h1_t, d_h2_t, dlabels_t = shuffler(d_h1_t, d_h2_t)
                        dlabels_t = Variable(dlabels_t).cuda()
                        d_input_t = torch.stack((d_h1_t, d_h2_t), 1)
                        d_output_t = discriminator(d_input_t).view(-1)
                        new_d_output_t = d_output_t > 0.5
                        d_total += len(new_d_output_t)
                        d_correct += len(new_d_output_t)- (
                            new_d_output_t ^ dlabels_t.byte()).sum().cpu().numpy()

                    # auxilary_classifier1 accuracy
                    t_pred = classifier(x1_t)
                    _, predicted = torch.max(t_pred.data, 1)
                    ac1_total += len(t_gt1)
                    ac1_correct += (predicted == t_gt1).sum().cpu().numpy()

                    cos = F.cosine_similarity(h1_t, h2_t, dim=1, eps=1e-6)
                    dist = F.relu((1-cos)*zsize/2)
                    cauchy_output = torch.reciprocal(dist+gamma)*gamma

                    try:
                        t_loss1 += loss_criterion(torch.squeeze(cauchy_output), s_t.float()).item()
                    except RuntimeError:
                        print(torch.squeeze(cauchy_output))
                        print(s_t)
                        print("s", torch.max(s_t.float()).item(),
                            torch.min(s_t.float()).item())
                        print("\nCO ", torch.max(torch.squeeze(cauchy_output)).item(),
                            torch.min(torch.squeeze(cauchy_output)).item())

                    if not len(t_indexes2) == int(batch_size/2):
                        cos = F.cosine_similarity(h1_t_new, h2_t_new, dim=1, eps=1e-6)
                        dist = F.relu((1-cos)*zsize/2)
                        cauchy_output = torch.reciprocal(dist+gamma)*gamma

                        try:
                            t_loss2 += loss_criterion(torch.squeeze(cauchy_output),
                                                    t_labels_2.float()).item()
                        except RuntimeError:
                            print(torch.squeeze(cauchy_output))
                            print(s_t)
                            print("s", torch.max(t_labels_2.float()).item(),
                                torch.min(t_labels_2.float()).item())
                            print("\nCO ", torch.max(torch.squeeze(cauchy_output)).item(),
                            torch.min(torch.squeeze(cauchy_output)).item())
                        t_run += 1

                    del t_input1
                    del t_input2
    return similarity_loss_dict, relational_loss_dict

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",
        required=False,
        help="Learning rate",
        default=0.0001,
        type = float)
    parser.add_argument("--checkpoint",
        required=False,
        help="Checkpoint model weight",
        default= None,
        type = str)
    parser.add_argument("--bs",
        required=False,
        default=256,
        help="Batchsize",
        type=int)
    parser.add_argument("--dpath",
        required=False,
        default = '/utils/dataset',
        help="Path to folder containing all data",
        type =str)
    parser.add_argument("--epochs",
        required=False,
        default=20,
        help="Number of epochs",
        type=int)
    parser.add_argument("--clscount",
        required=False,
        default=4,
        help="Number of classes",
        type=int)
    parser.add_argument("--spath",
        required=False,
        default = '/utils/model_weights',
        help="Path to folder in which models should be saved",
        type =str)
    parser.add_argument("--zsize",
        required=False,
        help="hash code length for the model",
        default=48,
        type=float)
    parser.add_argument("--alpha1",
        required=False,
        help="alpha1 for the model",
        default=0.5,
        type=float)
    parser.add_argument("--alpha2",
        required=False,
        help="alpha2 for the model.",
        default=1.0,
        type=float)
    parser.add_argument("--beta",
        required=False,
        help="Beta for the model.",
        default=0.5,
        type=float)
    parser.add_argument("--gamma",
        required=False,
        help="gamma for the model.",
        default=1,
        type=float)
    custom_args = parser.parse_args()

    main(custom_args)
