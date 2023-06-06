import argparse
import os
import os.path as osp

import numpy as np
from datumaro import DatasetItem, Polygon
from datumaro.components.project import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split

CLASSES = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field',
    'roundabout', 'harbor', 'swimming-pool', 'helicopter',
]


def load_dota_txt(txtfile):
    """Load DOTA's txt annotation.

    Args:
        txtfile (str): Filename of single txt annotation.

    Returns:
        dict: Annotation of single image.
    """
    gsd, polygons, labels, diffs = None, [], [], []
    if txtfile is None:
        pass
    elif not osp.isfile(txtfile):
        print(f"Can't find {txtfile}, treated as empty txtfile")
    else:
        with open(txtfile, 'r') as f:
            for line in f:
                if line.startswith('gsd'):
                    num = line.split(':')[-1]
                    try:
                        gsd = float(num)
                    except ValueError:
                        gsd = None
                    continue

                items = line.split(' ')
                if len(items) >= 9:
                    polygons.append([float(i) for i in items[:8]])
                    labels.append(items[8])
                    diffs.append(int(items[9]) if len(items) == 10 else 0)

    polygons = np.array(polygons, dtype=np.float32) if polygons else \
        np.zeros((0, 8), dtype=np.float32)
    diffs = np.array(diffs, dtype=np.int64) if diffs else \
        np.zeros((0,), dtype=np.int64)
    ann = dict(polygons=polygons, labels=labels, diffs=diffs)
    return dict(gsd=gsd, ann=ann)


def load_dota_single(imgfile, img_dir, ann_dir):
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
    txtfile = None if ann_dir is None else osp.join(ann_dir, img_id + '.txt')
    content = load_dota_txt(txtfile)

    content.update(
        dict(width=size[0], height=size[1], filename=imgfile, id=img_id))
    return content


def main(args):
    dsitems = []
    root_folder = args.data_root
    output_path = args.output

    for subset in ('train', 'val'):
        img_dir = osp.join(root_folder, subset, 'images')
        ann_dir = osp.join(root_folder, subset, 'labelTxt')
        img_list = os.listdir(img_dir)
        for imgfile in img_list:
            content = load_dota_single(imgfile, img_dir, ann_dir)
            polygons = []
            for polygon, label in zip(content['ann']['polygons'], content['ann']['labels']):
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
                        attributes={'filename': content['filename']}
                    )
                dsitems.append(dsitem)

    shuffled_indices = np.arange(len(dsitems))
    np.random.shuffle(shuffled_indices)
    test_indices = shuffled_indices[int(len(dsitems) * 0.8):]
    trainval_indices = shuffled_indices[:int(len(dsitems) * 0.8)]
    train_indices, val_indices = train_test_split(trainval_indices, test_size=0.25)
    for index in test_indices:
        dsitems[index].subset = 'test'
    for index in train_indices:
        dsitems[index].subset = 'train'
    for index in val_indices:
        dsitems[index].subset = 'val'
    dataset = Dataset.from_iterable(dsitems, categories=CLASSES)
    dataset.export(output_path, 'coco', default_image_ext='.jpg', save_media=True, image_ext='.jpg')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create COCO format annotations from DOTA dataset")
    parser.add_argument("--data-root", type=str, help="path to dota folder")
    parser.add_argument("--output", type=str, help="path to output folder")
    main(parser.parse_args())
