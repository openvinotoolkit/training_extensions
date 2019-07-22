import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from datasets import IBUG

from model.common import models_landmarks
from utils import landmarks_augmentation16
from utils.utils import save_model_cpu, load_model_state

data_root = "/home/share/landmark/image"
ld_root = "/home/share/landmark/mark"
PREVIEW_SIZE = 512

def draw_landmark_point(image, points):
    """
    Draw landmark point on image.
    """
    for point in points:
        cv2.circle(image, (int(point[0]), int(
            point[1])), 3, (0, 255, 0), -1, cv2.LINE_AA)
def test():
    dataset = IBUG(data_root, ld_root, test=True)

    dataset.transform = transforms.Compose([landmarks_augmentation16.Rescale((112, 112)),
                                        landmarks_augmentation16.ToTensor(switch_rb=True)])
    val_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)
    image_names = open(os.path.join(ld_root, 'test.txt'), 'r').read().splitlines()
    image_list = [i + '.jpg' for i in image_names]
    snap_name = os.path.join(os.getcwd(),'snapshots', 'LandNet_50500.pt')
    model = models_landmarks['mobilelandnet']()
    load_model_state(model, snap_name, -1, eval_state=True)
    for i, data in enumerate(val_loader, 0):
        data, gt_landmarks = data['img'], data['landmarks']
        predicted_landmarks = model(data)
        pts = np.reshape(predicted_landmarks.data.numpy(), (-1, 2))
        marks = pts * PREVIEW_SIZE
        img = cv2.imread(os.path.join(data_root,image_list[i]))
        img = cv2.resize(img, (PREVIEW_SIZE, PREVIEW_SIZE))
        draw_landmark_point(img, marks)
        cv2.imshow("result", img)
        if cv2.waitKey() == 27:
            exit()
def test_trans():
    dataset = IBUG(data_root, ld_root, test=True)
    dataset.transform = transforms.Compose([landmarks_augmentation16.RandomErasing(p=1.0),
                                            landmarks_augmentation16.Rescale((112, 112)),
                                            landmarks_augmentation16.ToTensor(switch_rb=True)])
    image_names = open(os.path.join(ld_root, 'test.txt'), 'r').read().splitlines()
    image_list = [i + '.jpg' for i in image_names]
    val_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)
    for i, data in enumerate(val_loader, 0):
        data, gt_landmarks = data['img'], data['landmarks']
        pts = np.reshape(gt_landmarks.data.numpy(), (-1, 2))
        marks = pts * 112
        img = data.mul(255).byte()
        img = img.numpy().squeeze(0).transpose((1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        draw_landmark_point(img, marks)
        origin = cv2.imread(os.path.join(data_root,image_list[i]))
        origin = cv2.resize(origin, (112, 112))
        cv2.imshow("origin", origin)
        cv2.imshow("trans", img)
        if cv2.waitKey() == 27:
            exit()

def test_image():
    root_dir = "/home/share/dsm_dataset/faces"
    snap_name = os.path.join(os.getcwd(),'snapshots', 'LandNet_50500.pt')
    model = models_landmarks['mobilelandnet']()
    load_model_state(model, snap_name, -1, eval_state=True)
    for root, _, files in os.walk(root_dir):
        for file in files:
            path = os.path.join(root, file)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img_resized = cv2.resize(img, (112, 112)).transpose(2, 0 ,1)
            data = (torch.from_numpy(img_resized).type(torch.FloatTensor) / 255).unsqueeze(0)
            predicted_landmarks = model(data)
            pts = np.reshape(predicted_landmarks.data.numpy(), (-1, 2))
            marks = pts * PREVIEW_SIZE
            img = cv2.resize(img, (PREVIEW_SIZE, PREVIEW_SIZE))
            draw_landmark_point(img, marks)
            cv2.imshow("result", img)
            if cv2.waitKey() == 27:
                exit()

def main():
    test_image()

if __name__ == '__main__':
    main()