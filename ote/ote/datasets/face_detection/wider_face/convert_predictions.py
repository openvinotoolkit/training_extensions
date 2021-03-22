import os

import mmcv
from tqdm import tqdm

from mmdet.datasets import build_dataset # pylint: disable=import-error


def convert_to_wider(config, results_file, out_folder, update_config):
    """ Main function. """

    if results_file is not None and not results_file.endswith(('.pkl', '.pickle')):
        raise ValueError('The input file must be a pkl file.')

    cfg = mmcv.Config.fromfile(config)
    if update_config:
        cfg.merge_from_dict(update_config)
    dataset = build_dataset(cfg.data.test)

    results = mmcv.load(results_file)

    wider_friendly_results = []
    for i, sample in enumerate(tqdm(dataset)):
        filename = sample['img_metas'][0].data['filename']
        folder, image_name = filename.split('/')[-2:]
        wider_friendly_results.append({'folder': folder, 'name': image_name[:-4],
                                       'boxes': results[i][0]})

    for result in wider_friendly_results:
        folder = os.path.join(out_folder, result['folder'])
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, result['name'] + '.txt'), 'w') as write_file:
            write_file.write(result['name'] + '\n')
            write_file.write(str(len(result['boxes'])) + '\n')
            for box in result['boxes']:
                box = box[0], box[1], box[2] - box[0], box[3] - box[1], box[4]
                write_file.write(' '.join([str(x) for x in box]) + '\n')
