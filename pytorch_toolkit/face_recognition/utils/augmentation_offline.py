import os
import cv2
import numpy as np
from tqdm import tqdm
import json




class Augumentor():
    def __init__(self, img_root, ld_root, save_dir, rot=False, flip=False, gray=False):
        self.images_root_path = img_root
        self.ld_root_path = ld_root
        self.landmarks_file = open(os.path.join(self.ld_root_path, 'test.txt'), 'r')
        self.rot = rot
        self.flip = flip
        self.gray = gray
        self.save_dir = save_dir

    def transfer(self):
        file_names = self.landmarks_file.readlines()
        for i in tqdm(range(len(file_names))):
            line = file_names[i].strip()

            name = line.split('/')[0]
            img_name = name + '.jpg'
            img_path = os.path.join(self.images_root_path, img_name)
            landmark_name = name + '.json'
            landmarks_path = os.path.join(self.ld_root_path, landmark_name)

            if self.gray:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            mark = np.array(json.load(open(landmarks_path, 'r')))

            if self.gray:
                gray_img = self.ToGary(img)
                save_name = self.save_dir + "/" + name + "_gray"
                print(save_name)
                cv2.imwrite(save_name + ".jpg", gray_img)
                points_to_save = mark.flatten()
                with open(save_name + ".json", mode='w') as file:
                    json.dump(list(points_to_save), file)
            if self.rot:
                for angle in range(-30, 31, 5):
                    rot_img, rot_mark = self.Rotate(img, mark, angle)
                    save_name = self.save_dir + "/" + name + "_rot" + str(angle)
                    print(save_name)
                    cv2.imwrite(save_name + ".jpg", rot_img)
                    points_to_save = rot_mark.flatten()
                    with open(save_name + ".json", mode='w') as file:
                        json.dump(list(points_to_save), file)

            if self.flip:
                flip_img, flip_mark = self.HorizontalFlip(img, mark)
                save_name = self.save_dir + "/" + name + "_flip"
                print(save_name)
                cv2.imwrite(save_name + ".jpg" ,flip_img)
                points_to_save = flip_mark.flatten()
                with open(save_name + ".json", mode='w') as file:
                    json.dump(list(points_to_save), file)

    def Rotate(self, image, landmarks, angle):

        h, w = image.shape[:2]
        rot_mat = cv2.getRotationMatrix2D((w*0.5, h*0.5), angle, 1.)
        image = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LANCZOS4)
        rot_mat_l = cv2.getRotationMatrix2D((0.5, 0.5), angle, 1.)
        landmarks = cv2.transform(landmarks.reshape(1, 16, 2), rot_mat_l).reshape(16, 2)
        return image, landmarks
    
    def HorizontalFlip(self, image, landmarks):

        image = cv2.flip(image, 1)
        landmarks = landmarks.reshape(16, 2)
        landmarks[:, 0] = 1. - landmarks[:, 0]
        return image, landmarks

    def ToGary(self, image):

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        return image
    
    def draw_landmark_point(self, image, points):
        h, w = image.shape[:2]
        pts = np.reshape(points, (-1, 2))
        pts = pts * h
        for point in pts:
            cv2.circle(image, (int(point[0]), int(
                point[1])), 3, (0, 255, 0), -1, cv2.LINE_AA) 

def test():
    path = "/home/vic/dataset/dsm_landmark/image/helen-trainset-22864555_1_flip_rot25.jpg"
    mark = "/home/vic/dataset/dsm_landmark/mark/helen-trainset-22864555_1_flip_rot25.json"
    img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2RGB)
    h = img.shape[0]
    print(img.shape)
    mark = np.array(json.load(open(mark, 'r')))
    pts = np.reshape(mark, (-1, 2))
    pts = pts * h
    print(pts)
    for point in pts:
        cv2.circle(img, (int(point[0]), int(
        point[1])), 3, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.imshow("img", img)
    cv2.waitKey()
def main():
    img_root = "/home/vic/dataset/dsm_landmark/image"
    ld_root = "/home/vic/dataset/dsm_landmark/mark"
    save_dir = "/home/vic/dataset/dsm_landmark/aug"
    aug = Augumentor(img_root, ld_root, save_dir, gray=True)
    aug.transfer()

if __name__ == "__main__":
    test()