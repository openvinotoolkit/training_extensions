import cv2 as cv
import glob
import os
import numpy as np
from pathlib import Path

TARGET_HEIGHT = 96
TARGET_WIDTH = 2000

OUTPUT_DIR='data/'
os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)
images = glob.glob('*.jpg')

def pad_image(image, size):
    h, w = image.shape[:2]
    if h != size[1] or w != size[0]:
        image = np.pad(image, ((0, size[1] - h), (0, size[0] - w)),
                               mode='constant', constant_values=255)
    return image

def main():
    for image in images:
        image_raw = cv.imread(image, cv.IMREAD_GRAYSCALE)
        img_h, img_w = image_raw.shape[0:2]
        scale = TARGET_HEIGHT / img_h
        width = int(scale * img_w)
        # resize to same height while keeping aspect ratio
        image_raw = cv.resize(image_raw, (width, TARGET_HEIGHT), interpolation=cv.INTER_AREA).astype(np.float32)
        # add padding to the right edge
        img = pad_image(image_raw, (TARGET_WIDTH, TARGET_HEIGHT))
        st = cv.imwrite(OUTPUT_DIR + image, img)
        print(image, st)


if __name__ == '__main__':
    main()

