import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm as tq

def list_apppend(list1,split_list=[], id=True):
    list2 = []
    for i,fname in enumerate(list1):
        p_id = fname.split('_')[1]
        if id:
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
    id_list = list_apppend(image_list, id=True)

    print(f'Total no. of unique patients: {len(id_list)}')

    train_threshold = 0.6*len(id_list)
    test_threshold = 0.2*len(id_list)

    for i,f_id in enumerate(id_list):
        if 0 < i < train_threshold:
            train_id_list.append(f_id)
        elif train_threshold < i < train_threshold+test_threshold:
            test_id_list.append(f_id)
        else:
            train_id_list.append(f_id)

    print(f'Total no. of unique patients in Train set: {len(train_id_list)}')
    print(f'Total no. of unique patients in Test set: {len(test_id_list)}')

    train_split = list_apppend(image_list, split_list=train_id_list, id=False)
    test_split = list_apppend(image_list, split_list=test_id_list, id=False)

    return train_split, test_split

def prepare_data_array(img_dir, mass_dir, split):
    data_arr = []
    b_flag, m_flag = False, False
    count_a,count_b = 0,0
    for case in tq(split):
        img_path = os.path.join(img_dir, case)
        mask_ben_path = os.path.join(mass_dir,'mass_ben_mask', case)
        mask_mlgn_path = os.path.join(mass_dir,'mass_mlgn_mask', case)
        img = cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), (320, 640), interpolation=cv2.INTER_CUBIC)
        mask_ben = cv2.resize(cv2.imread(mask_ben_path, cv2.IMREAD_GRAYSCALE), (320, 640), interpolation=cv2.INTER_CUBIC)
        mask_mlgn = cv2.resize(cv2.imread(mask_mlgn_path, cv2.IMREAD_GRAYSCALE), (320, 640), interpolation=cv2.INTER_CUBIC)

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

def prepare_data_npy(img_dir, mass_dir, train_split, test_split):
    train_data = prepare_data_array(img_dir, mass_dir, split=train_split)
    print('Train: ', len(train_data))
    np.save('data_splits/segmenter_train.npy', train_data)
    del train_data

    test_data = prepare_data_array(img_dir, mass_dir, split=test_split)
    print('Test: ', len(test_data))
    np.save('data_splits/segmenter_test.npy', test_data)
    del test_data


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_pth', type=str, required=True, help='Directory containing the arranged Dataset Directory')
    args = parser.parse_args()
    data_pth=args.data_pth

    img_dir = os.path.join(data_pth,'img')
    mass_dir = os.path.join(data_pth,'new_annotations/mass_mask')

    data_split = data_partition(img_dir)
    np.save('data_splits/train_split.npy',data_split[0])
    np.save('data_splits/test_split.npy',data_split[1])

    prepare_data_npy(img_dir, mass_dir, data_split[0], data_split[1])
