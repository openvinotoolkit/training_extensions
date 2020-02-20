import argparse
import json
import os

from PIL import Image
from tqdm import tqdm


def parse_wider_gt(ann_file):
    annotation = dict()
    with open(ann_file) as f:
        content = [line.strip() for line in f.readlines()]
        new_file = True
        i = 0
        while True:
            if new_file:
                image_name = content[i]
                annotation[image_name] = list()
                new_file = False
                i += 1
            else:
                bbox_num = int(content[i])
                i += 1
                for _ in range(bbox_num):
                    x, y, w, h = [int(x) for x in content[i].split(' ')[:4]]
                    if w >= 0 and h >= 0:
                        annotation[image_name].append([x, y, w, h])
                    else:
                        print('Ignored because of invalid size: ', [x, y, w, h])
                    i += 1
                if i == len(content):
                    break
                new_file = True

    return annotation


def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument('input_annotation',
                        help="Path to annotation file like wider_face_train_bbx_gt.txt")
    parser.add_argument('images_dir',
                        help="Path to folder with images like WIDER_train/images")
    parser.add_argument('output_annotation', help="Path to output json file")

    return parser.parse_args()


def convert_wider_annots(ann_file, data_dir, out_file):
    img_id = 0
    ann_id = 0
    cat_id = 1

    ann_dict = {}
    categories = [{"id": 1, "name": 'face'}]
    images = []
    annotations = []
    wider_annot_dict = parse_wider_gt(ann_file)

    for filename in tqdm(wider_annot_dict.keys()):
        image = {}
        image['id'] = img_id
        img_id += 1
        im = Image.open(os.path.join(data_dir, filename))
        image['width'] = im.height
        image['height'] = im.width
        image['file_name'] = os.path.relpath(
            os.path.join(data_dir, filename), os.path.dirname(out_file))
        images.append(image)

        for gt_bbox in wider_annot_dict[filename]:
            ann = {}
            ann['id'] = ann_id
            ann_id += 1
            ann['image_id'] = image['id']
            ann['segmentation'] = []
            ann['category_id'] = cat_id  # 1:"face" for WIDER
            ann['iscrowd'] = 0
            ann['area'] = gt_bbox[2] * gt_bbox[3]
            ann['bbox'] = gt_bbox
            annotations.append(ann)

    ann_dict['images'] = images
    ann_dict['categories'] = categories
    ann_dict['annotations'] = annotations
    print("Num categories: %s" % len(categories))
    print("Num images: %s" % len(images))
    print("Num annotations: %s" % len(annotations))
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    with open(out_file, 'w') as outfile:
        outfile.write(json.dumps(ann_dict))


if __name__ == '__main__':
    args = parse_args()
    convert_wider_annots(args.input_annotation, args.images_dir,
                         args.output_annotation)
