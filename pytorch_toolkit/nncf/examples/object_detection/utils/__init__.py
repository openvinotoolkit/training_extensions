from pathlib import Path

import cv2
import yaml
import sys

def mkdir_if_not_exists(path):
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)


def showAnnotation(dataset, class_names=None):
    for image, target in dataset:
        for anno in target:
            xmin = int(anno[0])
            ymin = int(anno[1])
            xmax = int(anno[2])
            ymax = int(anno[3])
            if class_names:
                name = class_names[int(anno[4])]
                print(name)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0))

        cv2.imshow("img", image)
        k = cv2.waitKey()
        if k == 27:
            sys.exit()


def read_config(path_to_config, img_size):
    with open(str(path_to_config)) as f:
        config = yaml.load(f)['CONFIG']
        return config[img_size]
