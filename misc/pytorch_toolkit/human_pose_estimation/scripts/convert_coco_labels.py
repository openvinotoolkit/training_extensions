import argparse
import json


def convert(labels_path, output_name):
    with open(labels_path, 'r') as f:
        labels = json.load(f)

    converted_labels = {'annotations': []}
    for annotation in labels['annotations']:
        if annotation['num_keypoints'] == 0:
            continue
        if annotation['iscrowd']:
            continue
        converted_annotation = {
            'bbox': annotation['bbox'],
            'keypoints': annotation['keypoints'],
            'image_path': '{:012}.jpg'.format(int(annotation['image_id']))
        }
        converted_labels['annotations'].append(converted_annotation)

    with open(output_name, 'w') as f:
        json.dump(converted_labels, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels-path', type=str, required=True, help='path to MS COCO labels file')
    parser.add_argument('--output-name', type=str, required=True, help='name of converted file')
    args = parser.parse_args()

    convert(args.labels_path, args.output_name)

