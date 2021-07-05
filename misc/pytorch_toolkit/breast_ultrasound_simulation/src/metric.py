import cv2
import numpy as np
import os
from tqdm import tqdm
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', required=True, help='path to directory')
args = parser.parse_args()



def filter(s, lf, hf):
    mask = np.zeros((s, s))
    c = np.array([s/2, s/2])
    
    for i in range(s):
        for j in range(s):
            n = np.linalg.norm(c - np.array([i, j]))
            if(n <hf and n>lf):
                mask[i][j] = 1

    return mask

def fft(img, mask, v):
    img = img/255
    m = np.mean(img)
    img = (img - m)
    f = np.fft.fft2(img)
    f = np.fft.fftshift(f)
    f = mask*f
    s = np.sum(np.abs(f)**2)

    return s/m,f


DIR = [args.dir]

# DIR.sort()
print(DIR)


sam_avg = [0]*len(DIR)
fft_avg = [0]*len(DIR)
s = 128
lf = s/8
hf = s/4
mask = filter(s, lf, hf)
for j, D in enumerate(tqdm(DIR)):
    images = [f for f in os.listdir(DIR[j])]
    for i, im in enumerate(images):
        
        a = cv2.imread(os.path.join(D, im),0)
        fa = fft(a, mask, hf - lf)
        sam_avg[j] += fa[0]
        
        fft_avg[j] = (fft_avg[j]*i + fa[1])/(i+1)

si =len(images)


sam_avg  = np.array(sam_avg)/si
print(sam_avg)

