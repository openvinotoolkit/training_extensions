import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', required=True, help='path to directory')
args = parser.parse_args()



def filterr(ss, lff, hff):
    maskk = np.zeros((ss, ss))
    c = np.array([ss/2, ss/2])

    for ii in range(ss):
        for jj in range(ss):
            n = np.linalg.norm(c - np.array([ii, jj]))
            if lff < n < hff:
                maskk[ii][jj] = 1

    return maskk

def fft(img, maskk, v):
    img = img/255
    m = np.mean(img)
    img = (img - m)
    f = np.fft.fft2(img)
    f = np.fft.fftshift(f)
    f = maskk*f
    ss = np.sum(np.abs(f)**2)

    return ss/m, f


dirr = [args.dir]

print(dirr)


sam_avg = [0]*len(dirr)
fft_avg = [0]*len(dirr)
s = 128
lf = s/8
hf = s/4
mask = filterr(s, lf, hf)
for j, D in enumerate(tqdm(dirr)):
    images = os.listdir(dirr[j])
    for i, im in enumerate(images):

        a = cv2.imread(os.path.join(D, im), 0)
        fa = fft(a, mask, hf - lf)
        sam_avg[j] += fa[0]

        fft_avg[j] = (fft_avg[j]*i + fa[1])/(i+1)

si = len(images)

sam_avg = np.array(sam_avg)/si
print(sam_avg)
