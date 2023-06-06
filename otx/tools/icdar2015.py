import argparse
import os
import os.path as osp

import numpy as np
from datumaro import DatasetItem, Polygon
from datumaro.components.project import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split

CLASSES = ['text']


def load_icdar_txt(txtfile):
    """Load DOTA's txt annotation.

    Args:
        txtfile (str): Filename of single txt annotation.

    Returns:
        dict: Annotation of single image.
    """
    polygons, labels = [], []
    if txtfile is None:
        pass
    elif not osp.isfile(txtfile):
        print(f"Can't find {txtfile}, treated as empty txtfile")
    else:
        with open(txtfile, 'r', encoding='utf-8-sig') as f:
            for line in f:
                items = line.split(',')
                if len(items) >= 9:
                    polygons.append([float(i) for i in items[:8]])
                    labels.append(items[8])

    polygons = np.array(polygons, dtype=np.float32) if polygons else \
        np.zeros((0, 8), dtype=np.float32)
    return dict(polygons=polygons, labels=labels)


def load_icdar_single(imgfile, img_dir, ann_dir):
    """Load DOTA's single image.

    Args:
        imgfile (str): Filename of single image.
        img_dir (str): Path of images.
        ann_dir (str): Path of annotations.

    Returns:
        dict: Content of single image.
    """
    img_id, ext = osp.splitext(imgfile)
    if ext not in ['.jpg', '.JPG', '.png', '.tif', '.bmp']:
        return None

    imgpath = osp.join(img_dir, imgfile)
    size = Image.open(imgpath).size
    txtfile = None if ann_dir is None else osp.join(ann_dir, 'gt_' + img_id + '.txt')
    content = load_icdar_txt(txtfile)

    content.update(
        dict(width=size[0], height=size[1], filename=imgfile, id=img_id))
    return content


def main(args):
    dsitems = []
    root_folder = args.data_root
    output_path = args.output

    img_dir = osp.join(root_folder, 'ch4_training_images')
    ann_dir = osp.join(root_folder, 'ch4_training_localization_transcription_gt')
    img_list = os.listdir(img_dir)
    train_indices, val_indices = train_test_split(range(len(img_list)), test_size=0.4)
    for idx in range(len(img_list)):
        imgfile = img_list[idx]
        content = load_icdar_single(imgfile, img_dir, ann_dir)
        polygons = []
        subset = 'train' if idx in train_indices else 'val'
        for polygon, label in zip(content['polygons'], content['labels']):
            # DOES NOT SUPPORT TEXT RECOGNITION
            label = 'text'
            polygon = Polygon(
                points=polygon,
                label=CLASSES.index(label),
                z_order=0,
                attributes={'filename': content['filename']}
            )
            polygons.append(polygon)
        if len(polygons) > 0:
            dsitem = DatasetItem(
                    id=content['id'],
                    annotations=polygons,
                    image=osp.join(img_dir, content['filename']),
                    attributes={'filename': content['filename']},
                    subset=subset
                )
            dsitems.append(dsitem)

    test_img_dir = osp.join(root_folder, 'ch4_test_images')
    test_ann_dir = osp.join(root_folder, 'Challenge4_Test_Task1_GT')
    test_img_list = os.listdir(test_img_dir)
    for idx in range(len(test_img_list)):
        imgfile = test_img_list[idx]
        content = load_icdar_single(imgfile, test_img_dir, test_ann_dir)
        polygons = []
        for polygon, label in zip(content['polygons'], content['labels']):
            # DOES NOT SUPPORT TEXT RECOGNITION
            label = 'text'
            polygon = Polygon(
                points=polygon,
                label=CLASSES.index(label),
                z_order=0,
                attributes={'filename': content['filename']}
            )
            polygons.append(polygon)
        if len(polygons) > 0:
            dsitem = DatasetItem(
                    id=content['id'],
                    annotations=polygons,
                    image=osp.join(test_img_dir, content['filename']),
                    attributes={'filename': content['filename']},
                    subset='test'
                )
            dsitems.append(dsitem)

    dataset = Dataset.from_iterable(dsitems, categories=CLASSES)
    dataset.export(output_path, 'coco', default_image_ext='.jpg', save_media=True, image_ext='.jpg')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create COCO format annotations from ICDAR2015 dataset")
    parser.add_argument("--data-root", type=str, help="path to dota folder")
    parser.add_argument("--output", type=str, help="path to output folder")
    main(parser.parse_args())
