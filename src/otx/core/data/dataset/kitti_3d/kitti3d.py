# flake8: noqa
# mypy: ignore-errors

import os

import numpy as np
from PIL import Image, ImageFile
from torch.utils import data

ImageFile.LOAD_TRUNCATED_IMAGES = True

import otx.core.data.dataset.kitti_3d.kitti_eval_python.kitti_common as kitti
from otx.core.data.dataset.kitti_3d.kitti_utils import (
    Calibration,
    PhotometricDistort,
    affine_transform,
    angle2class,
    get_affine_transform,
    get_objects_from_label,
)


class KITTI_Dataset(data.Dataset):
    def __init__(
        self,
        root_dir,
        split,
        use_3d_center=True,
        class_merging=False,
        use_dontcare=False,
        writelist=["Car"],
        bbox2d_type="anno",
        clip_2d=False,
        aug_pd=True,
        aug_crop=True,
        random_flip=0.5,
        random_crop=0.5,
        scale=0.05,
        shift=0.05,
        depth_scale="normal",
    ):
        # basic configuration
        self.root_dir = root_dir
        if split == "test":
            split = "val"
        self.split = split
        self.num_classes = 3
        self.max_objs = 50
        self.class_name = ["Pedestrian", "Car", "Cyclist"]
        self.cls2id = {"Pedestrian": 0, "Car": 1, "Cyclist": 2}
        self.resolution = np.array([1280, 384])  # W * H
        self.use_3d_center = use_3d_center
        self.writelist = writelist
        # anno: use src annotations as GT, proj: use projected 2d bboxes as GT
        self.bbox2d_type = bbox2d_type
        assert self.bbox2d_type in ["anno", "proj"]
        self.meanshape = False
        self.class_merging = class_merging
        self.use_dontcare = use_dontcare

        if self.class_merging:
            self.writelist.extend(["Van", "Truck"])
        if self.use_dontcare:
            self.writelist.extend(["DontCare"])

        # data split loading
        assert self.split in ["train", "val", "test"]
        self.split_file = os.path.join(self.root_dir, "ImageSets", self.split + ".txt")
        self.idx_list = [x.strip() for x in open(self.split_file).readlines()]

        # path configuration
        self.data_dir = os.path.join(self.root_dir, "training")
        self.image_dir = os.path.join(self.data_dir, "image_2")
        self.calib_dir = os.path.join(self.data_dir, "calib")
        self.label_dir = os.path.join(self.data_dir, "label_2")

        # data augmentation configuration
        self.data_augmentation = True if split in ["train"] else False
        self.data_augmentation = False

        self.aug_pd = aug_pd
        self.aug_crop = aug_crop
        self.aug_calib = False

        self.random_flip = random_flip
        self.random_crop = random_crop
        self.scale = scale
        self.shift = shift

        self.depth_scale = depth_scale

        # statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.cls_mean_size = np.array(
            [
                [1.76255119, 0.66068622, 0.84422524],
                [1.52563191462, 1.62856739989, 3.88311640418],
                [1.73698127, 0.59706367, 1.76282397],
            ],
        )
        if not self.meanshape:
            self.cls_mean_size = np.zeros_like(self.cls_mean_size, dtype=np.float32)

        # others
        self.downsample = 32
        self.pd = PhotometricDistort()
        self.clip_2d = clip_2d

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, "%06d.png" % idx)
        assert os.path.exists(img_file), f"{img_file} does not exist!"
        return Image.open(img_file)  # (H, W, 3) RGB mode

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, "%06d.txt" % idx)
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, "%06d.txt" % idx)
        assert os.path.exists(calib_file)
        return Calibration(calib_file)

    def prepare_kitti_format(self):
        img_ids = [int(self.idx_list[item]) for item in range(self.__len__())]
        return kitti.get_label_annos(self.label_dir, img_ids)

    def __len__(self):
        return self.idx_list.__len__()

    def __getitem__(self, item):
        #  ============================   get inputs   ===========================
        index = int(self.idx_list[item])  # index mapping, get real data id
        # image loading
        img = self.get_image(index)
        img_size = np.array(img.size)
        features_size = self.resolution // self.downsample  # W * H

        # data augmentation for image
        center = np.array(img_size) / 2
        crop_size, crop_scale = img_size, 1
        random_flip_flag = False
        if self.data_augmentation:
            if self.aug_pd:
                img = np.array(img).astype(np.float32)
                img = self.pd(img).astype(np.uint8)
                img = Image.fromarray(img)

            if np.random.random() < self.random_flip:
                random_flip_flag = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            if self.aug_crop:
                if np.random.random() < self.random_crop:
                    crop_scale = np.clip(np.random.randn() * self.scale + 1, 1 - self.scale, 1 + self.scale)
                    crop_size = img_size * crop_scale
                    center[0] += img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                    center[1] += img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)

        # add affine transformation for 2d images.
        trans, trans_inv = get_affine_transform(center, crop_size, 0, self.resolution, inv=1)
        img = img.transform(
            tuple(self.resolution.tolist()),
            method=Image.AFFINE,
            data=tuple(trans_inv.reshape(-1).tolist()),
            resample=Image.BILINEAR,
        )

        # image encoding
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # C * H * W -> (384 * 1280)

        info = {"img_id": index, "img_size": img_size, "bbox_downsample_ratio": img_size / features_size}

        #  ============================   get labels   ==============================
        objects = self.get_label(index)
        calib = self.get_calib(index)
        # data augmentation for labels
        if random_flip_flag:
            if self.aug_calib:
                calib.flip(img_size)
            for object in objects:
                [x1, _, x2, _] = object.box2d
                object.box2d[0], object.box2d[2] = img_size[0] - x2, img_size[0] - x1
                object.alpha = np.pi - object.alpha
                object.ry = np.pi - object.ry
                if self.aug_calib:
                    object.pos[0] *= -1
                if object.alpha > np.pi:
                    object.alpha -= 2 * np.pi  # check range
                if object.alpha < -np.pi:
                    object.alpha += 2 * np.pi
                if object.ry > np.pi:
                    object.ry -= 2 * np.pi
                if object.ry < -np.pi:
                    object.ry += 2 * np.pi

        # labels encoding
        calibs = np.zeros((self.max_objs, 3, 4), dtype=np.float32)
        indices = np.zeros((self.max_objs), dtype=np.int64)
        mask_2d = np.zeros((self.max_objs), dtype=bool)
        labels = np.zeros((self.max_objs), dtype=np.int8)
        depth = np.zeros((self.max_objs, 1), dtype=np.float32)
        heading_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
        heading_res = np.zeros((self.max_objs, 1), dtype=np.float32)
        size_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
        size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        boxes = np.zeros((self.max_objs, 4), dtype=np.float32)
        boxes_3d = np.zeros((self.max_objs, 6), dtype=np.float32)
        objects_list = []

        object_num = len(objects) if len(objects) < self.max_objs else self.max_objs

        for i in range(object_num):
            # filter objects by writelist
            if objects[i].cls_type not in self.writelist:
                continue

            # filter inappropriate samples
            if objects[i].level_str == "UnKnown" or objects[i].pos[-1] < 2:
                continue

            # ignore the samples beyond the threshold [hard encoding]
            threshold = 65
            if objects[i].pos[-1] > threshold:
                continue

            # process 2d bbox & get 2d center
            bbox_2d = objects[i].box2d.copy()

            # add affine transformation for 2d boxes.
            bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
            bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)

            # process 3d center
            center_2d = np.array(
                [(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2],
                dtype=np.float32,
            )  # W * H
            corner_2d = bbox_2d.copy()

            center_3d = objects[i].pos + [0, -objects[i].h / 2, 0]  # real 3D center in 3D space
            center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
            center_3d, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
            center_3d = center_3d[0]  # shape adjustment
            if random_flip_flag and not self.aug_calib:  # random flip for center3d
                center_3d[0] = img_size[0] - center_3d[0]
            center_3d = affine_transform(center_3d.reshape(-1), trans)
            affine_transform(objects[i].pos, trans)
            # filter 3d center out of img
            proj_inside_img = True

            if center_3d[0] < 0 or center_3d[0] >= self.resolution[0]:
                proj_inside_img = False
            if center_3d[1] < 0 or center_3d[1] >= self.resolution[1]:
                proj_inside_img = False

            if proj_inside_img == False:
                continue

            # class
            cls_id = self.cls2id[objects[i].cls_type]
            labels[i] = cls_id

            # encoding 2d/3d boxes
            w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
            size_2d[i] = 1.0 * w, 1.0 * h

            center_2d_norm = center_2d / self.resolution
            size_2d_norm = size_2d[i] / self.resolution

            corner_2d_norm = corner_2d
            corner_2d_norm[0:2] = corner_2d[0:2] / self.resolution
            corner_2d_norm[2:4] = corner_2d[2:4] / self.resolution
            center_3d_norm = center_3d / self.resolution

            l, r = center_3d_norm[0] - corner_2d_norm[0], corner_2d_norm[2] - center_3d_norm[0]
            t, b = center_3d_norm[1] - corner_2d_norm[1], corner_2d_norm[3] - center_3d_norm[1]

            if l < 0 or r < 0 or t < 0 or b < 0:
                if self.clip_2d:
                    l = np.clip(l, 0, 1)
                    r = np.clip(r, 0, 1)
                    t = np.clip(t, 0, 1)
                    b = np.clip(b, 0, 1)
                else:
                    continue

            boxes[i] = center_2d_norm[0], center_2d_norm[1], size_2d_norm[0], size_2d_norm[1]
            boxes_3d[i] = center_3d_norm[0], center_3d_norm[1], l, r, t, b

            # encoding depth
            if self.depth_scale == "normal":
                depth[i] = objects[i].pos[-1] * crop_scale

            elif self.depth_scale == "inverse":
                depth[i] = objects[i].pos[-1] / crop_scale

            elif self.depth_scale == "none":
                depth[i] = objects[i].pos[-1]

            # encoding heading angle
            heading_angle = calib.ry2alpha(objects[i].ry, (objects[i].box2d[0] + objects[i].box2d[2]) / 2)
            if heading_angle > np.pi:
                heading_angle -= 2 * np.pi  # check range
            if heading_angle < -np.pi:
                heading_angle += 2 * np.pi
            heading_bin[i], heading_res[i] = angle2class(heading_angle)

            # encoding size_3d
            src_size_3d[i] = np.array([objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)
            mean_size = self.cls_mean_size[self.cls2id[objects[i].cls_type]]
            size_3d[i] = src_size_3d[i] - mean_size

            if objects[i].trucation <= 0.5 and objects[i].occlusion <= 2:
                mask_2d[i] = 1
                objects_list.append(objects[i])

            calibs[i] = calib.P2

        # collect return data
        inputs = img
        targets = {
            "labels": labels[mask_2d],
            "boxes": boxes[mask_2d],
            "boxes_3d": boxes_3d[mask_2d],
            "depth": depth[mask_2d],
            "size_2d": size_2d[mask_2d],
            "size_3d": size_3d[mask_2d],
            "heading_bin": heading_bin[mask_2d],
            "heading_res": heading_res[mask_2d],
        }

        info = {"img_id": index, "img_size": img_size, "bbox_downsample_ratio": img_size / features_size}

        return inputs, calib, targets, info, objects_list


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    cfg = {
        "root_dir": "../../../data/KITTI",
        "random_flip": 0.0,
        "random_crop": 1.0,
        "scale": 0.8,
        "shift": 0.1,
        "use_dontcare": False,
        "class_merging": False,
        "writelist": ["Pedestrian", "Car", "Cyclist"],
        "use_3d_center": False,
    }
    dataset = KITTI_Dataset("train", cfg)
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    print(dataset.writelist)

    for batch_idx, (inputs, targets, info) in enumerate(dataloader):
        # test image
        img = inputs[0].numpy().transpose(1, 2, 0)
        img = (img * dataset.std + dataset.mean) * 255
        img = Image.fromarray(img.astype(np.uint8))
        img.show()
        # print(targets['size_3d'][0][0])

        # test heatmap
        heatmap = targets["heatmap"][0]  # image id
        heatmap = Image.fromarray(heatmap[0].numpy() * 255)  # cats id
        heatmap.show()

        break

    # print ground truth fisrt
    objects = dataset.get_label(0)
    for object in objects:
        print(object.to_kitti_format())
