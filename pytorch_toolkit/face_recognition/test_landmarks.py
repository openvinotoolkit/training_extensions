import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from datasets import IBUG

from model.common import models_landmarks
from utils import landmarks_augmentation16, face_detector
from utils.pts_tools import box_in_image, move_box, get_square_box
from utils.utils import save_model_cpu, load_model_state

data_root = "/home/vic/dataset/dsm_landmark/image"
ld_root = "/home/vic/dataset/dsm_landmark/mark"
PREVIEW_SIZE = 512

def draw_landmark_point(image, points):
    """
    Draw landmark point on image.
    """
    for point in points:
        cv2.circle(image, (int(point[0]), int(
            point[1])), 1, (0, 255, 0), -1, cv2.LINE_AA)
def read_img():
    data_root = "/home/vic/dataset/landmark_dataset/ibug"
    snap_name = os.path.join(os.getcwd(),'snapshots', 'LandNet_45500.pt')
    model = models_landmarks['mobilelandnet']()
    load_model_state(model, snap_name, -1, eval_state=True)
    # ld_root = "/home/vic/dataset/dsm_landmark/mark"
    # image_names = open(os.path.join(ld_root, 'test.txt'), 'r').read().splitlines()
    image_names = open('ibug.txt', 'r').read().splitlines()
    image_list = [i + '.jpg' for i in image_names]
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
    # for name in image_list:
    #     path = os.path.join(data_root, name)
    #     img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        h, w, _ = img.shape
        conf, boxes = face_detector.get_facebox(img)
        if boxes is None:
            continue
        left_top = []
        square_boxes = []
        faces = []
        for box in boxes:
            if box_in_image(box, img):
                # Move down
                diff_height_width = (box[3] - box[1]) - (box[2] - box[0])
                offset_y = int(abs(diff_height_width / 2))
                box_moved = move_box(box, [0, offset_y])
                # Make box square.
                square_box = get_square_box(box_moved)
                square_boxes.append(square_box)
                # Save for transform
                left_top.append([square_box[0], square_box[1], square_box[2] - square_box[0], square_box[3] - square_box[1]])
                face = cv2.resize(img[square_box[1]:square_box[3], square_box[0]:square_box[2]], (112, 112)).transpose(2, 0, 1)
                faces.append(face[np.newaxis, :])
    
        if len(faces) == 0:
            continue
        if len(faces) > 1:
            data = np.vstack(faces)
        else:
            data = faces[0]
        data = (torch.from_numpy(data).type(torch.FloatTensor) / 255)
        predicted_landmarks = model(data)
        pts = np.reshape(predicted_landmarks.data.numpy(), (-1, 16, 2))
        for i in range(pts.shape[0]): 
            marks = pts[i] * np.array(left_top[i][2:4]).reshape(1, 2) + np.array(left_top[i][0:2]).reshape(1, 2)
            draw_landmark_point(img, marks)
        # face_detector.draw_result(img, conf, boxes)
        # marks = pts * PREVIEW_SIZE
        # img = cv2.resize(img, (PREVIEW_SIZE, PREVIEW_SIZE))
        # draw_landmark_point(img, marks)
        face_detector.draw_box(img, square_boxes)
        cv2.imshow("result", img)
        if cv2.waitKey(1) == 27:
            exit()
                
def test():

    dataset = IBUG(data_root, ld_root, test=True)

    dataset.transform = transforms.Compose([landmarks_augmentation16.Rescale((112, 112)),
                                        landmarks_augmentation16.ToTensor(switch_rb=True)])
    val_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)
    image_names = open(os.path.join(ld_root, 'test_gray.txt'), 'r').read().splitlines()
    image_list = [i + '.jpg' for i in image_names]
    snap_name = os.path.join(os.getcwd(),'snapshots', 'LandNet_45500.pt')
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
    snap_name = os.path.join(os.getcwd(),'snapshots', 'LandNet_45500.pt')
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
def test_video():
    snap_name = os.path.join(os.getcwd(),'snapshots', 'LandNet_45500.pt')
    model = models_landmarks['mobilelandnet']()
    load_model_state(model, snap_name, -1, eval_state=True)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        # print(frame)
        img_resized = cv2.resize(frame, (112, 112)).transpose(2, 0 ,1)
        data = (torch.from_numpy(img_resized).type(torch.FloatTensor) / 255).unsqueeze(0)
        predicted_landmarks = model(data)
        pts = np.reshape(predicted_landmarks.data.numpy(), (-1, 2))
        marks = pts * PREVIEW_SIZE
        img = cv2.resize(frame, (PREVIEW_SIZE, PREVIEW_SIZE))
        draw_landmark_point(frame, marks)
        cv2.imshow("result", frame)
        if cv2.waitKey(10) == 27:
            exit()
def main():
    test()

if __name__ == '__main__':
    main()