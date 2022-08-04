import argparse
import numpy as np
import json
import os


DATA_DIR = '/home/yuchunli/git/training_extensions/vitens_dataset/Vitens-Legionella-coco'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script to build detection coco annotation files for class incremental learning')
    parser.add_argument('--mode', type=str, help='train|val', default='train')
    parser.add_argument('--output', type=str, required=True, help='Path to destination to build')
    parser.add_argument('--seed', type=int, help='Seed value', default=1)
    parser.add_argument('--num_of_subset', type=int, default=-1, help='number of images')
    return parser.parse_args()


def prepare_vitens_data(
    output_dir,
    seed=1,
    mode='train',
    num_of_subset=-1,
):
    def _save_anno(name, images, annotations, path):
        print(f'>> Processing data {name}.json saved ({len(images)} images {len(annotations)} annotations)')
        new_anno = {}
        new_anno['images'] = list(images)
        new_anno['annotations'] = list(annotations)
        new_anno['licenses'] = anno['licenses']
        new_anno['categories'] = anno['categories']
        new_anno['info'] = anno['info']
        if not os.path.exists(path):
            os.mkdir(path)
        
        with open(f'{path}/{name}.json', 'w') as f:
            json.dump(new_anno, f)
        print(f'>> Data {name}.json saved ({len(images)} images {len(annotations)} annotations)')

    np.random.seed(seed)
    # Get split data info (train | val)
    ANNODIR = os.path.join(DATA_DIR, 'annotations')
    tmp_mode = mode
    if mode == 'test':
        tmp_mode = 'val'
    anno = json.load(open(os.path.join(ANNODIR, f'instances_{tmp_mode}.json')))

    images = np.array(anno['images'])  # coco all images
    if num_of_subset <= 0:
        num_of_subset = len(images)
    selected_img_indices = np.random.choice(range(len(images)), size=num_of_subset, replace=False)
    selected_id = []
    selected_imgs = images[selected_img_indices]
    for selected_img in selected_imgs:
        selected_id.append(selected_img['id'])

    total_ann = np.array(anno['annotations'])
    selected_ann = []

    for ann in total_ann:
        if ann['image_id'] in selected_id:
            selected_ann.append(ann)
    new_save_name = f'instances_{mode}_seed{seed}_{num_of_subset}'
    _save_anno(new_save_name, selected_imgs, selected_ann, output_dir)


# def main(seeds=[1, 10, 100, 1234, 4321], num_images=[12, 24, 36]):
#     for seed in seeds:
#         for num_img in num_images:
#             prepare_vitens_data(DATA_DIR, seed, num_of_subset=num_img)


if __name__ == '__main__':
    args = parse_args()
    prepare_vitens_data(
        output_dir=args.output,
        seed=args.seed,
        mode=args.mode,
        num_of_subset=args.num_of_subset
    )
