import numpy as np
import os
import cv2
import json
from tqdm import tqdm as tq
import matplotlib.pyplot as plt
import random
from collections import defaultdict

def generate_patchlist(save_path,patchtype,fold_no=0):
    """Generates positive slices in each fold

    Parameters
    ----------
    save_path: str
        Folder location where jsons are to be stored
    fold_no: int
        Integer between 0-9 that specifies the fold
    category: str
        Positive/negative

    Returns
    -------
    None

    """

    with open(save_path+'fold'+str(fold_no)+'_pos_neg_eq.json') as file:
        j_data = json.load(file)
    with open(save_path+'/'+patchtype+'_slices.json') as c:
        pos_slices_json=json.load(c)

    # print(pos_slices_json)

    train_set = j_data['train_set']
    valid_set = j_data['valid_set']
    test_set = j_data['test_set']
    train_seg_list = []
    val_seg_list = []
    test_seg_list = []

    for i in tq(train_set):
        if i in pos_slices_json:
            train_seg_list.append(i)

    for i in tq(valid_set):
        if i in pos_slices_json:
            val_seg_list.append(i)

    for i in tq(test_set):
        if i in pos_slices_json:
            test_seg_list.append(i)

    patch_npy={}
    patch_npy = defaultdict(lambda:[],patch_npy)
    patch_npy['train_set'] = train_seg_list
    patch_npy['valid_set'] = val_seg_list
    patch_npy['test_set'] = test_seg_list

    with open(save_path+'/'+patchtype+'_patchlist_f'+str(fold_no)+'.json', 'w') as z:
        json.dump(patch_npy,z)



def generate_negative_patch(jsonpath,fold,data_path,lung_segpath,savepath,category='train_set'):
    """Gereates patches which doesn't have nodules

    Parameters
    ----------
    jsonpath: str
        Folder location where json files are stored
    fold: int
        Fold number
    category: str
        train_set/val_set/test_set
    data_path: str
        Folder location where img numpy arrays are stored
    lung_segpath: str
        Folder location where lung segmentation mask is stored
    savepath: strr
        Folder location to save the generated patches

    Returns
    -------
    None
    """

    imgpath = data_path + '/img'

    with open(jsonpath+'negative_patchlist_f'+str(fold)+'.json') as file:
        j_data = json.load(file)

    img_dir = imgpath
    mask_dir = lung_segpath
    nm_list = j_data[category]

    size = 64
    index = 0
    for img_name in tq(nm_list):
        #Loading the masks as uint8 as threshold function accepts 8bit image as parameter.
        img = np.load(os.path.join(img_dir, img_name)).astype(np.float32)#*255
        mask = np.load(os.path.join(mask_dir, img_name)).astype(np.uint8)#*255

        if np.any(mask):
            #Convert grayscale image to binary
            _, th_mask = cv2.threshold(mask, 0.5, 1, 0,cv2.THRESH_BINARY) #parameters are ip_img,threshold,max_value
            contours, hierarchy = cv2.findContours(th_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda x: cv2.contourArea(x))

            #In certain cases there could be more than 2 contour, hence taking the largest 2 which will be lung
            contours = contours[1:]


            for cntr in contours:
                patch_count = 2
                for _ in range(patch_count):
                    xr,yr,wr,hr = cv2.boundingRect(cntr) #Gives X,Y cordinate of BBox origin,height and width
                    # xc,yc = xr+wr/2,yr+hr/2

                    try:

                        x, y = random.randrange(xr, xr+wr-size/2),random.randrange(yr, yr+hr-size/2)

                    except:
                        prob = random.randrange(0, 1)
                        if prob>0.5:
                            x, y = random.randrange(xr, xr+wr/2),random.randrange(yr, yr+hr/2)
                        else:
                            x, y = random.randrange(int(xr+wr/2),xr+wr),random.randrange(int(yr+hr/2),yr+hr)

                    if x+size<512 & y+size<512:
                        patch_img = img[y: y+size, x: x+size].copy().astype(np.float16)
                        patch_mask = np.zeros((size,size)).astype(np.float16)

                    else:
                        if x-size<=0 & y-size<=0:
                            patch_img = img[0: size, 0: size].copy().astype(np.float16)
                            patch_mask = np.zeros((size,size)).astype(np.float16)

                        elif x-size<=0 & y-size>0:
                            patch_img = img[y-size: y, 0: size].copy().astype(np.float16)
                            patch_mask = np.zeros((size,size)).astype(np.float16)

                        elif x-size>0 & y-size<=0:
                            patch_img = img[0: size, x-size: x].copy().astype(np.float16)
                            patch_mask = np.zeros((size,size)).astype(np.float16)

                        else:

                            patch_img = img[y-size: y, x-size: x].copy().astype(np.float16)
                            patch_mask = np.zeros((size,size)).astype(np.float16)



                    if np.shape(patch_img) != (64,64):
                        print('shape',np.shape(patch_img))
                        print('cordinate of patch',x,x+size,y,y+size)
                        print('cordinate of BBox',xr,yr,wr,hr)

                    index += 1
                    img_savepath = savepath+'/patches/'+'/img/'
                    mask_savepath = savepath+'/patches/'+'/mask/'
                    if not os.path.isdir(img_savepath):
                        os.makedirs(savepath+'/patches/'+'/img/')
                        np.save(img_savepath+'patch_'+str(fold)+'_'+str(index)+'.npy',patch_img)
                    else:
                        np.save(img_savepath+'patch_'+str(fold)+'_'+str(index)+'.npy',patch_img)

                    if not os.path.isdir(mask_savepath):
                        os.makedirs(savepath+'/patches/'+'/mask/')
                        np.save(mask_savepath+'patch_'+str(fold)+'_'+str(index)+'.npy',patch_mask)
                    else:
                        np.save(mask_savepath+'patch_'+str(fold)+'_'+str(index)+'.npy',patch_mask)


def generate_positive_patch(jsonpath,fold,data_path,savepath,category='train_set'):
    """Generate patches with nodules

    Parameters
    ----------
    jsonpath: str
        Folder location where json files are stored
    fold: int
        Fold number
    category: str
        train_set/val_set/test_set
    data_path: str
        Folder location which has folder img and mask

    savepath: strr
        Folder location to save the generated patches

    Returns
    -------
    None

    """
    imgpath = data_path + '/img/'
    maskpath = data_path + '/mask/'

    with open(jsonpath+'/positive_patchlist_f'+str(fold)+'.json') as file:
        j_data = json.load(file)

    img_dir = imgpath
    mask_dir = maskpath
    nm_list = j_data[category]

    size = 64
    index = 0
    for img_name in tq(nm_list):
        #Loading the masks as uint8 as threshold function accepts 8bit image as parameter.
        img = np.load(os.path.join(img_dir, img_name)).astype(np.float16)
        mask = np.load(os.path.join(mask_dir, img_name))/255
        mask = mask.astype(np.uint8)

        if np.any(mask):
            #Convert grayscale image to binary
            _, th_mask = cv2.threshold(mask, 0.5, 1, 0,cv2.THRESH_BINARY) #parameters are ip_img,threshold,max_value
            contours, _ = cv2.findContours(th_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda x: cv2.contourArea(x))

            for cntr in contours:
                patch_count = 4

                xr,yr,wr,hr = cv2.boundingRect(cntr) #Gives X,Y cordinate of BBox origin,height and width
                xc,yc = int(xr+wr/2),int(yr+hr/2)

                if int(yc-size/2) <0 or int(xc-size/2)<0:
                    if int(yc-size/2) <0 and int(xc-size/2)<0:
                        patch_img1 = img[0:size , 0:size].copy().astype(np.float16)
                        patch_mask1 = mask[0:size , 0:size].copy().astype(np.float16)

                    elif int(yc-size/2) >0 and int(xc-size/2)<0:
                        patch_img1 = img[int(yc-size/2):int(yc+size/2) , 0:size].copy().astype(np.float16)
                        patch_mask1 = mask[int(yc-size/2):int(yc+size/2) , 0:size].copy().astype(np.float16)

                    elif int(yc-size/2) <0 and int(xc-size/2)>0:
                        patch_img1 = img[0:size ,int(xc-size/2):int(xc+size/2)].copy().astype(np.float16)
                        patch_mask1 = mask[0:size ,int(xc-size/2):int(xc+size/2)].copy().astype(np.float16)


                elif int(yc+size/2)>512 or int(xc+size/2)>512:
                    if int(yc+size/2)>512 and int(xc+size/2)>512:
                        m = yc+size - 512
                        n = xc + size - 512
                        patch_img1 = img[int(yc-m):512,int(xc-n):512].copy().astype(np.float16)
                        patch_mask1 = mask[int(yc-m):512,int(xc-n):512].copy().astype(np.float16)

                    elif int(yc+size/2)>512 and int(xc+size/2)<512:
                        m = yc+size - 512
                        patch_img1 = img[int(yc-m):512,int(xc-size/2):int(xc+size/2)].copy().astype(np.float16)
                        patch_mask1 = mask[int(yc-m):512,int(xc-size/2):int(xc+size/2)].copy().astype(np.float16)

                    elif int(yc+size/2)<512 and int(xc+size/2)>512:
                        n = xc+size - 512
                        patch_img1 = img[int(yc-size/2):int(yc+size/2),int(xc-n):512].copy().astype(np.float16)
                        patch_mask1 = mask[int(yc-size/2):int(yc+size/2),int(xc-n):512].copy().astype(np.float16)

                elif (int(yc-size/2)>=0 and int(yc+size/2)<=512) :
                     if(int(xc-size/2)>=0 and int(xc+size/2)<=512):
                        patch_img1 = img[
                            int(yc-size/2):int(yc+size/2),
                            int(xc-size/2):int(xc+size/2)].copy().astype(np.float16)
                        patch_mask1 = mask[
                            int(yc-size/2):int(yc+size/2),
                            int(xc-size/2):int(xc+size/2)].copy().astype(np.float16)

                if np.shape(patch_img1) != (64,64):
                    print('shape',np.shape(patch_img1))
                    print('cordinate of patch',x,x+size,y,y+size)
                    print('cordinate of BBox',xr,yr,wr,hr)

                img_savepath = savepath+'/patches/'+category+'/img/'
                mask_savepath = savepath+'/patches/'+category+'/mask/'
                if not os.path.isdir(img_savepath):
                    os.makedirs(savepath+'/patches/'+category+'/img/')
                    np.save(img_savepath+'patch_'+str(fold)+'_'+str(index)+'.npy',patch_img1)
                else:
                    np.save(img_savepath+'patch_'+str(fold)+'_'+str(index)+'.npy',patch_img1)

                if not os.path.isdir(mask_savepath):
                    os.makedirs(savepath+'/patches/'+category+'/mask/')
                    np.save(mask_savepath+'patch_'+str(fold)+'_'+str(index)+'.npy',patch_mask1)
                else:
                    np.save(mask_savepath+'patch_'+str(fold)+'_'+str(index)+'.npy',patch_mask1)

                index += 1
                for i in range(patch_count):
                    xc,yc = xr,yr
                    xc,yc = xr+wr,yr+hr

                    if i == 0:

                        if xc+size<512 and yc+size<512:
                            patch_img = img[yc:yc+size,xc:xc+size].copy().astype(np.float16)
                            patch_mask = mask[yc:yc+size,xc:xc+size].copy().astype(np.float16)

                        elif xc+size>512 and yc+size<512:
                            m = xc+size-512
                            patch_img = img[yc:yc+size,xc-m:xc+size-m].copy().astype(np.float16)
                            patch_mask = mask[yc:yc+size,xc-m:xc+size-m].copy().astype(np.float16)

                        elif xc+size<512 and yc+size>512:
                            n = yc+size-512
                            patch_img = img[yc-n:yc+size-n,xc:xc+size].copy().astype(np.float16)
                            patch_mask = mask[yc-n:yc+size-n,xc:xc+size].copy().astype(np.float16)
                        else:
                            m = xc+size-512
                            n = yc+size-512
                            patch_img = img[yc-n:yc+size-n,xc-m:xc+size-m].copy().astype(np.float16)
                            patch_mask = mask[yc-n:yc+size-n,xc-m:xc+size-m].copy().astype(np.float16)
                    elif i ==1:

                        if xc-size>0 and yc+size<512:
                            patch_img = img[yc:yc+size,xc-size:xc].copy().astype(np.float16)
                            patch_mask = mask[yc:yc+size,xc-size:xc].copy().astype(np.float16)

                        elif xc-size<0 and yc+size<512:

                            patch_img = img[yc:yc+size,0:size].copy().astype(np.float16)
                            patch_mask = mask[yc:yc+size,0:size].copy().astype(np.float16)

                        elif xc-size>0 and yc+size>512:
                            n = yc+size-512

                            patch_img = img[yc-n:yc+size-n,xc-size:xc].copy().astype(np.float16)
                            patch_mask = mask[yc-n:yc+size-n,xc-size:xc].copy().astype(np.float16)

                        else:
                            n = yc+size-512

                            patch_img = img[yc-n:yc+size-n,0:size].copy().astype(np.float16)
                            patch_mask = mask[yc-n:yc+size-n,0:size].copy().astype(np.float16)
                    elif i ==2:

                        if xc+size<512 and yc-size>0:
                            patch_img = img[yc-size:yc,xc:xc+size].copy().astype(np.float16)
                            patch_mask = mask[yc-size:yc,xc:xc+size].copy().astype(np.float16)

                        elif xc+size>512 and yc-size>0:
                            m = xc+size-512
                            patch_img = img[yc-size:yc,xc-m:xc+size-m].copy().astype(np.float16)
                            patch_mask = mask[yc-size:yc,xc-m:xc+size-m].copy().astype(np.float16)

                        elif xc+size<512 and yc-size<0:
                            patch_img = img[0:size,xc:xc+size].copy().astype(np.float16)
                            patch_mask = mask[0:size,xc:xc+size].copy().astype(np.float16)

                        else:
                            m = xc+size-512
                            patch_img = img[yc-size:yc,xc-m:xc+size-m].copy().astype(np.float16)
                            patch_mask = mask[yc-size:yc,xc-m:xc+size-m].copy().astype(np.float16)

                    elif i==3:

                        if xc-size>0 and yc-size>0:
                            patch_img = img[yc-size:yc,xc-size:xc].copy().astype(np.float16)
                            patch_mask = mask[yc-size:yc,xc-size:xc].copy().astype(np.float16)

                        elif xc-size<0 and yc-size>0:
                            m = xc+size-512
                            patch_img = img[yc-size:yc,0:size].copy().astype(np.float16)
                            patch_mask = mask[yc-size:yc,0:size].copy().astype(np.float16)

                        elif xc-size>0 and yc-size<0:
                            patch_img = img[0:size,xc-size:xc].copy().astype(np.float16)
                            patch_mask = mask[0:size,xc-size:xc].copy().astype(np.float16)

                        else:
                            patch_img = img[0:size,0:size].copy().astype(np.float16)
                            patch_mask = mask[0:size,0:size].copy().astype(np.float16)


                    if np.shape(patch_img) != (64,64):
                        print('shape',np.shape(patch_img))

                    img_savepath = savepath+'/patches/'+category+'/img/'
                    mask_savepath = savepath+'/patches/'+category+'/mask/'
                    if not os.path.isdir(img_savepath):
                        os.makedirs(savepath+'/patches/'+category+'/img/')
                        np.save(img_savepath+'patch_'+str(fold)+'_'+str(index)+'.npy',patch_img)
                    else:
                        np.save(img_savepath+'patch_'+str(fold)+'_'+str(index)+'.npy',patch_img)

                    if not os.path.isdir(mask_savepath):
                        os.makedirs(savepath+'/patches/'+category+'/mask/')
                        np.save(mask_savepath+'patch_'+str(fold)+'_'+str(index)+'.npy',patch_mask)
                    else:
                        np.save(mask_savepath+'patch_'+str(fold)+'_'+str(index)+'.npy',patch_mask)

                    index += 1
