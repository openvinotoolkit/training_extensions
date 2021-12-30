import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', required=True, help='path to directory')
args = parser.parse_args()

def filter_custom(req_shape, low_freq, high_freq):
    filtered_mask = np.zeros((req_shape, req_shape))
    c = np.array([req_shape/2, req_shape/2])
    for i_val in range(req_shape):
        for j_val in range(req_shape):
            n = np.linalg.norm(c - np.array([i_val, j_val]))
            if low_freq<n<high_freq:
                filtered_mask[i_val][j_val] = 1

    return filtered_mask

def fft(img, filt_mask, v):
    img = img/255
    mean = np.mean(img)
    img = (img - mean)
    f_img = np.fft.fft2(img)
    f_img = np.fft.fftshift(f_img)
    f_img = filt_mask*f_img
    sq_sum = np.sum(np.abs(f_img)**2)

    return sq_sum/mean,f_img


DIR = [args.dir]

# DIR.sort()
print(DIR)


sam_avg = [0]*len(DIR)
fft_avg = [0]*len(DIR)
SHAPE = 128
LOW_FREQ = SHAPE/8
HIGH_FREQ = SHAPE/4
mask = filter_custom(SHAPE, LOW_FREQ, HIGH_FREQ)
for j, D in enumerate(tqdm(DIR)):
    images = list(os.listdir(DIR[j]))
    for i, im in enumerate(images):
        a = cv2.imread(os.path.join(D, im),0)
        fa = fft(a, mask, HIGH_FREQ - LOW_FREQ)
        sam_avg[j] += fa[0]
        fft_avg[j] = (fft_avg[j]*i + fa[1])/(i+1)

si =len(images)
sam_avg  = np.array(sam_avg)/si
print(sam_avg)
