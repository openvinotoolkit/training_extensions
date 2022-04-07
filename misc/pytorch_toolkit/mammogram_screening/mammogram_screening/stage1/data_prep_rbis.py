import cv2
import numpy as np
import os
from tqdm import tqdm as tq
from ..train_utils.get_config import get_config
from ..train_utils.downloader import download_data

def list_apppend(list1, split_list, idx=True):
    list2 = []
    for fname in list1:
        p_id = fname.split('_')[1]
        if idx:
            if p_id not in list2:
                list2.append(p_id)
        else:
            if p_id in split_list:
                list2.append(fname)
    return list2

def data_partition(img_dir):
    train_id_list = []
    test_id_list = []
    train_split, test_split = [],[]
    image_list = os.listdir(img_dir)
    id_list = list_apppend(image_list,split_list=[], idx=True)
    print(f"Total no. of unique patients: {len(id_list)}")
    train_threshold = 0.5*len(id_list) # Thresholds should be changed as required
    test_threshold = 0.5*len(id_list)

    for i,f_id in enumerate(id_list):
        if 0 < i < train_threshold:
            train_id_list.append(f_id)
        elif train_threshold < i <= train_threshold+test_threshold+1:
            test_id_list.append(f_id)
        else:
            train_id_list.append(f_id)

    print(f'Total no. of unique patients in Train set: {len(train_id_list)}')
    print(f'Total no. of unique patients in Test set: {len(test_id_list)}')

    train_split = list_apppend(image_list, split_list=train_id_list, idx=False)
    test_split = list_apppend(image_list, split_list=test_id_list, idx=False)

    return train_split, test_split

def prepare_data_array(img_dir, mass_dir, split):
    data_arr = []
    b_flag, m_flag = False, False
    count_a,count_b = 0,0
    for case in tq(split):
        img_path = os.path.join(img_dir, case)
        mask_ben_path = os.path.join(mass_dir,'mass_ben_mask', case)
        mask_mlgn_path = os.path.join(mass_dir,'mass_mlgn_mask', case)
        img = cv2.resize(
                        cv2.imread(img_path, cv2.IMREAD_GRAYSCALE),
                        (320, 640), interpolation=cv2.INTER_CUBIC)
        mask_ben = cv2.resize(
                            cv2.imread(mask_ben_path, cv2.IMREAD_GRAYSCALE),
                            (320, 640), interpolation=cv2.INTER_CUBIC)
        mask_mlgn = cv2.resize(
                            cv2.imread(mask_mlgn_path, cv2.IMREAD_GRAYSCALE),
                            (320, 640), interpolation=cv2.INTER_CUBIC)

        if np.sum(mask_ben) > 0:
            b_flag = True
            count_a +=1
        if np.sum(mask_mlgn) > 0:
            m_flag = True
            count_b +=1

        if b_flag:
            mask = mask_ben
            cls = 0
            data_dict = {'img': img, 'mask': mask, 'cls': cls, 'file_name': case}
            data_arr.append(data_dict)
        if m_flag:
            mask = mask_mlgn
            cls = 1
            data_dict = {'img': img, 'mask': mask, 'cls': cls, 'file_name': case}
            data_arr.append(data_dict)
        if not b_flag and not m_flag:
            print(f'Skip case: {case}')

    return data_arr

def prepare_data_npy(data_path, save_path, train_split, test_split):

    img_dir = os.path.join(data_path,'images')
    mass_dir = os.path.join(data_path,'annotations/mass_mask')
    train_data = prepare_data_array(img_dir, mass_dir, split=train_split)
    print('Train: ', len(train_data))
    np.save(os.path.join(save_path,'segmenter_train.npy'), train_data)
    del train_data

    test_data = prepare_data_array(img_dir, mass_dir, split=test_split)
    print('Test: ', len(test_data))
    np.save(os.path.join(save_path,'segmenter_test.npy'), test_data)
    del test_data

def data_prep():
    config = get_config(action='data_prep',config_path='configs/', stage= 'stage1')
    data_path = config['data_path']
    save_path = config['save_path']
    if not os.path.exists(data_path):
        download_data()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img_dir = os.path.join(data_path,'images')
    data_split = data_partition(img_dir)
    train_split = os.path.join(save_path,'train_split.npy')
    test_split = os.path.join(save_path,'test_split.npy')
    np.save(train_split, data_split[0])
    np.save(test_split, data_split[1])
    prepare_data_npy(data_path, save_path, data_split[0], data_split[1])

if __name__ == '__main__':
    data_prep()
