import argparse
import os
import os.path as osp

import numpy as np
from datumaro import DatasetItem, Polygon
from datumaro.components.project import Dataset
from PIL import Image

CLASSES = ['license plate']


def load_ccpd_vertices(fname, height, width):
    """Load CCPD's vertices annotation.

    Args:
        txtfile (str): Filename of single txt annotation.

    Returns:
        dict: Annotation of single image.
    """
    polygons, labels = [], []
    fname = fname.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
    vertices = fname[3].split("_")
    polygon = []
    for vertice in vertices:
        vertice = vertice.split("&")
        x, y = float(vertice[0]), float(vertice[1])
        polygon.extend([x, y])
    polygons.append(polygon)
    labels.append(CLASSES[0])
    polygons = np.array(polygons, dtype=np.float32) if polygons else \
        np.zeros((0, 8), dtype=np.float32)
    return dict(polygons=polygons, labels=labels)


def load_ccpd_single(imgfile, img_dir):
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
    content = load_ccpd_vertices(imgfile, height=size[1], width=size[0])

    content.update(
        dict(width=size[0], height=size[1], filename=imgfile, id=img_id))
    return content


def main(args):
    dsitems = []
    root_folder = args.data_root
    output_path = args.output

    for subset in ('train', 'val', 'test'):
        img_dir = osp.join(root_folder, subset)
        img_list = os.listdir(img_dir)
        for imgfile in img_list:
            content = load_ccpd_single(imgfile, img_dir)
            polygons = []
            for polygon, label in zip(content['polygons'], content['labels']):
                # DOES NOT SUPPORT TEXT RECOGNITION
                label = 'license plate'
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

    dataset = Dataset.from_iterable(dsitems, categories=CLASSES)
    dataset.export(output_path, 'coco', default_image_ext='.jpg', save_media=True, image_ext='.jpg')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create COCO format annotations from ICDAR2015 dataset")
    parser.add_argument("--data-root", type=str, help="path to dota folder")
    parser.add_argument("--output", type=str, help="path to output folder")
    main(parser.parse_args())
