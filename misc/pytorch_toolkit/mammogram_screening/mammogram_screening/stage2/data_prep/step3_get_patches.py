import numpy as np
import cv2
from tqdm import tqdm as tq

def random_point(img):
    lower_bound = 256
    upper_bound = img.shape[0] - 256 - 64
    point = np.random.randint(lower_bound, upper_bound, 2) # upper left point
    x, y, w, h = point[0], point[1], 64, 64

    return (x, y, w, h)

def random_patch(img):
    while True:
        # Continue extracting random patches until a patch from the foregorund region is selected.
        # background has all zero pixel intensities
        (x, y, w, h) = random_point(img)
        if np.count_nonzero(img[y:y+h, x:x+w]) > 64*64 - 10:
            return (x, y, w, h)

def extract_bags(predictions):
    pad_top = 224
    pad_bottom = 224
    pad_left = 224
    pad_right = 224
    SIZE_PATCH = 64
    bags = []

    for i, item in tq(enumerate(predictions)):

        cls = item['cls']
        img = item['img']
        mask = item['mask']
        mask_pred = item['mask_pred']
        file_name = item['file_name']

        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        mask = cv2.copyMakeBorder(mask, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        mask_pred = cv2.copyMakeBorder(mask_pred, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

        mask_pred_thres = mask_pred.copy()

        thres = 10
        mask_pred_thres[mask_pred < thres] = 0
        mask_pred_thres[mask_pred >= thres] = 255

        kernel = np.ones((31,31), np.uint8)
        mask_pred_dilate = cv2.dilate(mask_pred, kernel, iterations=1)
        kernel = np.ones((31,31), np.uint8)
        mask_pred_dilate = cv2.erode(mask_pred_dilate, kernel, iterations=1)

        mask_pred_dilate[mask_pred_dilate < thres] = 0
        mask_pred_dilate[mask_pred_dilate >= thres] = 255


        ctrs, hier = cv2.findContours(mask_pred_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        for j, cnt in enumerate(ctrs):
            max_area = max(cv2.contourArea(cnt), max_area)

        patches = []

        for j, cnt in enumerate(ctrs):
            if cv2.contourArea(cnt) > 0.05*max_area and cv2.contourArea(cnt) > 200:
                xr,yr,wr,hr = cv2.boundingRect(cnt)

                x,y,w,h = xr,yr,wr,hr
                x1,y1,w1,h1 = x,y,SIZE_PATCH, SIZE_PATCH
                x2,y2,w2,h2 = x+w-SIZE_PATCH, y, SIZE_PATCH, SIZE_PATCH
                x3,y3,w3,h3 = x, y+h-SIZE_PATCH, SIZE_PATCH, SIZE_PATCH
                x4,y4,w4,h4 = x+w-SIZE_PATCH, y+h-SIZE_PATCH, SIZE_PATCH, SIZE_PATCH
                x5,y5,w5,h5 = x+(w//2)-(SIZE_PATCH//2), y+(h//2)-(SIZE_PATCH//2), SIZE_PATCH, SIZE_PATCH

                patches.append([img[y1:y1+h1, x1:x1+w1], img[y2:y2+h2, x2:x2+w2], img[y3:y3+h3, x3:x3+w3], img[y4:y4+h4, x4:x4+w4], img[y5:y5+h5, x5:x5+w5]])

        if len(patches) >= 1:
            d = {'patches': patches, 'cls': cls, 'file_name': file_name, 'random': False, 'has_mass': True}
            bags.append(d)
            print(i, len(patches), cls, file_name)
            print(len(bags))
        else:
            patches = []
            if cls == 0:
                for k in range(50):
                    (x,y,w,h) = random_patch(img)
                    patches.append([img[y:y+h, x:x+w]])
                d = {'patches': patches, 'cls': cls, 'file_name': file_name, 'random': True, 'has_mass': False}
                bags.append(d)
            else:
                for k in range(50):
                    (x,y,w,h) = random_patch(img)
                    patches.append([img[y:y+h, x:x+w]])
                d = {'patches': patches, 'cls': cls, 'file_name': file_name, 'random': True, 'has_mass': True}
                bags.append(d)

                
    return(bags)

if __name__ == '__main__':

    print('Preparing Validation Bags')
    predictions = np.load('prepared_data/mass_predictions/val_all_pred.npy', allow_pickle=True)
    print(len(predictions))
    bags=extract_bags(predictions)
    np.save('prepared_data/bags/val_bags_pred.npy', bags)
        
    print('Preparing Training Bags')
    predictions = np.load('prepared_data/mass_predictions/train_all_pred.npy', allow_pickle=True)
    print(len(predictions))
    bags=extract_bags(predictions)
    np.save('prepared_data/bags/train_bags_pred.npy', bags)
