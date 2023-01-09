import torch
import torch.nn.functional as F
import numpy as np


def shuffler(h1, h2):
    sh = torch.randint(0, 2, (h1.size(0),))  # 1 means keeping h1 and 0 means keeping h2
    #print(sh)
    h1new = torch.matmul(torch.diag(1-sh).float(), h2.cpu())+torch.matmul(
        torch.diag(sh).float(),h1.cpu())
    h2new = torch.matmul(torch.diag(1-sh).float(),h1.cpu())+torch.matmul(
        torch.diag(sh).float(),h2.cpu())
    #print(h1new.shape)
    #print(h2new.shape)
    return h1new.cuda(), h2new.cuda(), sh


def precision(q_class, ret_classes):
    initlist = [int(q_class == i) for i in ret_classes]
    #print(initlist)
    den = np.sum(initlist)
    #print(den)
    if den == 0:
        return 0
    x = 0
    preclist = [0]*len(initlist)
    for idx, pts in enumerate(initlist):
        x += pts #rel(n)
        preclist[idx] = x/(idx+1) #rel(n)/k
    #print(preclist)
    num = np.dot(preclist, initlist)
    #print(num)
    #print(num/den)
    return num/den


def imagenormalize(img):
    # print(img.shape)
    img = img.astype(np.float32)  # converting array of ints to floats
    img_a = img[0, :, :]
    img_b = img[1, :, :]
    img_c = img[2, :, :]  # Extracting single channels from 3 channel image
    # normalizing per channel data:
    img_a = (img_a - np.min(img_a)) / (np.max(img_a) - np.min(img_a))
    img_b = (img_b - np.min(img_b)) / (np.max(img_b) - np.min(img_b))
    img_c = (img_c - np.min(img_c)) / (np.max(img_c) - np.min(img_c))

    # putting the 3 channels back together:
    img_norm = np.empty(img.shape, dtype=np.float32)
    img_norm[0, :, :] = img_a
    img_norm[1, :, :] = img_b
    img_norm[2, :, :] = img_c
    return img_norm


def normaliser(x):
    minx = torch.min(x)
    maxx = torch.max(x)
    normalised_vector = (x-minx) / (maxx-minx)
    return normalised_vector


def rescale_vector(width, vec, sf):
    count = width
    x = torch.unsqueeze(vec, 0)
    while count > 0:
        x = torch.cat((x, x), 0)
        count -= 1
    x_plus = torch.unsqueeze(x, 0)
    x_plus = torch.cat((x_plus, x_plus, x_plus), 0)
    x_plus = F.interpolate(x_plus, scale_factor=sf)
    return x_plus

#print("a")

def re_classes(sorted_pool,q_name):
    value = []
    for i, _ in enumerate(sorted_pool):
        if (q_name.split("_")[0]+'_'+q_name.split("_")[1]) ==(
            sorted_pool[i][0].split("_")[0]+'_'+sorted_pool[i][0].split("_")[1]):
            value.append(2)
        elif (q_name.split("_")[0])== (
                sorted_pool[i][0].split("_")[0] and q_name.split("_")[1])!= (
                sorted_pool[i][0].split("_")[1]):
            value.append(1)
        else:
            value.append(0)
    value2 = sorted(value, reverse=True)
    return value, value2
def discounted_cumulative_gain(result):
    dcg = []
    for idx, val in enumerate(result):
        numerator = 2**val - 1
        # add 2 because python 0-index
        denominator =  np.log2(idx + 2)
        score = numerator/denominator
        dcg.append(score)
    return sum(dcg)
