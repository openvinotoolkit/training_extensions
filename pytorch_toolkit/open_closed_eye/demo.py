
import argparse
import os
import os.path as osp
import argparse
import glog as log
import cv2 as cv
import numpy as np
from utils.ie_tools import load_ie_model

def parse_args():
    parser = argparse.ArgumentParser(description='Open close eyes classifier')
    parser.add_argument('model')
    parser.add_argument('data_root')    
    args = parser.parse_args()
    return args

def load_eye_db(root_dir):
    data = []
    for subdir, dirs, files in os.walk(root_dir):
        for i, file in enumerate(files):                
            full_path = os.path.join(subdir, file)
            state = 1 if file[0] == 'o' else 0
            data.append({'filename' : full_path, 'label' : state})
    return data

def main():
    args = parse_args()

    test_db = load_eye_db(args.data_root)
    net = load_ie_model(args.model, 'CPU', None)
    _, _, height, width = net.get_input_shape().shape

    for sample in test_db:
        img = cv.imread(sample['filename'])
        assert not img is None
        h,w,_ = img.shape
        out = net.forward(cv.resize(img, (width, height)))
        isOpen = out[0][0][0][0] < out[0][1][0][0]
        if isOpen:
            cv.rectangle(img,(1,1), (w-1,h-1), (0,255,0),2)
        else:
            cv.rectangle(img,(1,1), (w-1,h-1), (0,0,255),2)
        cv.imshow("Eye", img)        
        cv.waitKey(0)

if __name__ == '__main__':
    main()
