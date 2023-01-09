import itertools
import os
import random
import numpy as np
from PIL import Image
from torch.utils import data
def dataset(root):
    if not os.path.exists(root):
        raise Exception("Invalid path destination")
    if root.split("/")[-1]=="train":
        im_thresh = 5000 #optimum 5000
        type2_num = 30000 #optimum 200000
        type1_num = 30000#optimum 100000
        type0_num = 30000 #optimum 100000
    elif root.split("/")[-1] in ['val', "test1"]:
        im_thresh = 2000 #optimum 1500
        type0_num = 10000 #optimum 10000
        type1_num = 10000 #optimum 10000
        type2_num = 10000 #optimum 10000

    classes = list(os.listdir(root))
    classes = sorted(classes)
    cls_to_idx = {classes[i]:i for i in range(len(classes))}
    print(classes)
    print(cls_to_idx)
    #cls_num_imgs = {cls_to_idx[i]:len(os.listdir(os.path.join(fpath,i))) for i in classes}
    #print(cls_num_imgs)
    images1 = []
    images2 = []
    dataset2 = []
    for cls_name in classes:
        images_temp = [(im,cls_name) for im in os.listdir(os.path.join(root,cls_name))]
        random.shuffle(images_temp)
        #images.extend(images_temp)
        images1.extend(images_temp[:200])
        images2 = images_temp[:min(im_thresh,len(images_temp))]
        dataset2.extend(list(itertools.combinations(images2,2)))

    #dataset2.extend(list(itertools.combinations(images,2)))
    dataset1 = list(itertools.combinations(images1,2))
    #print(len(images_temp))
    count = [0]*3
    newdataset = []
    for data1 in dataset1:
        #print(data[1][1])
        if not data1[0][1] == data1[1][1] :
            tag = 2
            count[2]+=1
            img_path1 = os.path.join(os.path.join(root,data1[0][1]), data1[0][0])
            img_path2 = os.path.join(os.path.join(root,data1[1][1]), data1[1][0])
            item = (img_path1,img_path2,tag,cls_to_idx[data1[0][1]],cls_to_idx[data1[1][1]])
            newdataset.append(item)
            if count[2] == type2_num:
                break

    random.shuffle(dataset2)
    #print(dataset2)
    for data1 in dataset2:
        img_path1 = os.path.join(os.path.join(root,data1[0][1]), data1[0][0])
        img_path2 = os.path.join(os.path.join(root,data1[1][1]), data1[1][0])
        if (((data1[0][0].split("_")[0]+'_' + data1[0][0].split("_")[1])
        == (data1[1][0].split("_")[0]+'_' + data1[1][0].split("_")[1]))):
            tag = 0
            if count[0] < type0_num:
            #if (t==0):
                count[0]+=1
                item = (img_path1,img_path2,tag,cls_to_idx[data1[0][1]],cls_to_idx[data1[1][1]])
                # item = (data[0][0],data[1][0],t)
                newdataset.append(item)
        else:
            tag = 1
            if count[1] < type1_num :
            #if (t==1):
                count[1]+=1
                item = (img_path1,img_path2,tag,cls_to_idx[data1[0][1]],cls_to_idx[data1[1][1]])
                newdataset.append(item)
    random.shuffle(newdataset)
    #print("Count",count)
    #print(newdataset[5])
    return newdataset

#fpath = "/storage/asim/Hashing_MedMNISTV2/test1"
#newdataset = dataset(fpath)
def find_classes(root):
    classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

class DataloderImg(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        classes, class_to_idx = find_classes(root)
        samples = dataset(root)
        self.root = root
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        path1, path2, target, target1, target2 = self.samples[index]
        img1 = np.load(path1)
        img2 = np.load(path2)
        sample1 = Image.fromarray(img1)
        #print(sample1)
        sample2 = Image.fromarray(img2)
        if self.transform is not None:
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)

        if self.target_transform is not None:
            target = self.target_transform(target)
            target1 = self.target_transform(target1)
            target2 = self.target_transform(target2)
            #c cluster_centers = self.target_transform(cluster_centers)
        return sample1, sample2, target , target1, target2 # , cluster_centers
    def __len__(self):
        return len(self.samples)
