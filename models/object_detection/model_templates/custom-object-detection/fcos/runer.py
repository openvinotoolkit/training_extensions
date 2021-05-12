from subprocess import run
import yaml

def load_map(path):
    with open(path) as f:
        metrics = yaml.load(f)['metrics']
        map = [v['value'] for v in metrics if v['key'] == 'bbox'][0]

    return float(map) / 100

datasets = [
    {
        'name': 'Aerial_tiled',
        'train-ann-file': '/media/cluster_fs/datasets/object_detection/Aerial_Maritime/tiled/train/_annotations.coco.json',
        'train-data-root': '/media/cluster_fs/datasets/object_detection/Aerial_Maritime/tiled/train/',
        'val-ann-file': '/media/cluster_fs/datasets/object_detection/Aerial_Maritime/tiled/valid/_annotations.coco.json',
        'val-data-root': '/media/cluster_fs/datasets/object_detection/Aerial_Maritime/tiled/valid/',
        'test-ann-file': '/media/cluster_fs/datasets/object_detection/Aerial_Maritime/tiled/test/_annotations.coco.json',
        'test-data-root': '/media/cluster_fs/datasets/object_detection/Aerial_Maritime/tiled/test/'
    },
    {
        'name': 'BBCD',
        'train-ann-file': '/media/cluster_fs/datasets/object_detection/BBCD/train/_annotations.coco.json',
        'train-data-root': '/media/cluster_fs/datasets/object_detection/BBCD/train/',
        'val-ann-file': '/media/cluster_fs/datasets/object_detection/BBCD/valid/_annotations.coco.json',
        'val-data-root': '/media/cluster_fs/datasets/object_detection/BBCD/valid/',
        'test-ann-file': '/media/cluster_fs/datasets/object_detection/BBCD/test/_annotations.coco.json',
        'test-data-root': '/media/cluster_fs/datasets/object_detection/BBCD/test/'
    },
    {
        'name': 'Pothole',
        'train-ann-file': '/media/cluster_fs/datasets/object_detection/Pothole/train/_annotations.coco.json',
        'train-data-root': '/media/cluster_fs/datasets/object_detection/Pothole/train/',
        'val-ann-file': '/media/cluster_fs/datasets/object_detection/Pothole/valid/_annotations.coco.json',
        'val-data-root': '/media/cluster_fs/datasets/object_detection/Pothole/valid/',
        'test-ann-file': '/media/cluster_fs/datasets/object_detection/Pothole/test/_annotations.coco.json',
        'test-data-root': '/media/cluster_fs/datasets/object_detection/Pothole/test'
    },
    {
        'name': 'Wildfire',
        'train-ann-file': '/media/cluster_fs/datasets/object_detection/Wildfire_smoke/train/_annotations.coco.json',
        'train-data-root': '/media/cluster_fs/datasets/object_detection/Wildfire_smoke/train/',
        'val-ann-file': '/media/cluster_fs/datasets/object_detection/Wildfire_smoke/valid/_annotations.coco.json',
        'val-data-root': '/media/cluster_fs/datasets/object_detection/Wildfire_smoke/valid/',
        'test-ann-file': '/media/cluster_fs/datasets/object_detection/Wildfire_smoke/test/_annotations.coco.json',
        'test-data-root': '/media/cluster_fs/datasets/object_detection/Wildfire_smoke/test'
    },
    {
        'name': 'Vitens-tiled',
        'train-ann-file': '',
        'train-data-root': '',
        'val-ann-file': '',
        'val-data-root': '',
        'test-ann-file': '',
        'test-data-root': ''
    },
    {
        'name': 'Fish',
        'train-ann-file': '/media/cluster_fs/datasets/object_detection/kbts-fish-coco/annotations/instances_train.json',
        'train-data-root': '/media/cluster_fs/datasets/object_detection/kbts-fish-coco/images/train/',
        'val-ann-file': '/media/cluster_fs/datasets/object_detection/kbts-fish-coco/annotations/instances_val.json',
        'val-data-root': '/media/cluster_fs/datasets/object_detection/kbts-fish-coco/images/val/',
        'test-ann-file': '/media/cluster_fs/datasets/object_detection/kbts-fish-coco/annotations/instances_test.json',
        'test-data-root': '/media/cluster_fs/datasets/object_detection/kbts-fish-coco/images/test/'
    },
    {
        'name': 'PCD',
        'train-ann-file': '/media/cluster_fs/datasets/object_detection/pcd-coco/annotations/instances_train.json',
        'train-data-root': '/media/cluster_fs/datasets/object_detection/pcd-coco/images/train/',
        'val-ann-file': '/media/cluster_fs/datasets/object_detection/pcd-coco/annotations/instances_val.json',
        'val-data-root': '/media/cluster_fs/datasets/object_detection/pcd-coco/images/val/',
        'test-ann-file': '/media/cluster_fs/datasets/object_detection/pcd-coco/annotations/instances_val.json',
        'test-data-root': '/media/cluster_fs/datasets/object_detection/pcd-coco/images/val/',
    },
    {
        'name': 'Weed',
        'train-ann-file': '/media/cluster_fs/datasets/object_detection/weed-coco/annotations/instances_train.json',
        'train-data-root': '/media/cluster_fs/datasets/object_detection/weed-coco/images/train/',
        'val-ann-file': '/media/cluster_fs/datasets/object_detection/weed-coco/annotations/instances_val.json',
        'val-data-root': '/media/cluster_fs/datasets/object_detection/weed-coco/images/val/',
        'test-ann-file': '/media/cluster_fs/datasets/object_detection/weed-coco/annotations/instances_val.json',
        'test-data-root': '/media/cluster_fs/datasets/object_detection/weed-coco/images/val/'
    },
    {
        'name': 'DIOPSIS',
        'train-ann-file': '/media/cluster_fs/datasets/object_detection/diopsis_coco/annotations/instances_train.json',
        'train-data-root': '/media/cluster_fs/datasets/object_detection/diopsis_coco/images/train/',
        'val-ann-file': '/media/cluster_fs/datasets/object_detection/diopsis_coco/annotations/instances_val.json',
        'val-data-root': '/media/cluster_fs/datasets/object_detection/diopsis_coco/images/val/',
        'test-ann-file': '/media/cluster_fs/datasets/object_detection/diopsis_coco/annotations/instances_val.json',
        'test-data-root': '/media/cluster_fs/datasets/object_detection/diopsis_coco/images/val/'
    },
    {
        'name': 'Aerial_large',
        'train-ann-file': '/media/cluster_fs/datasets/object_detection/Aerial_Maritime/large/train/_annotations.coco.json',
        'train-data-root': '/media/cluster_fs/datasets/object_detection/Aerial_Maritime/large/train/',
        'val-ann-file': '/media/cluster_fs/datasets/object_detection/Aerial_Maritime/large/valid/_annotations.coco.json',
        'val-data-root': '/media/cluster_fs/datasets/object_detection/Aerial_Maritime/large/valid/',
        'test-ann-file': '/media/cluster_fs/datasets/object_detection/Aerial_Maritime/large/test/_annotations.coco.json',
        'test-data-root': '/media/cluster_fs/datasets/object_detection/Aerial_Maritime/large/test/'
    },
    {
        'name': 'Dice',
        'train-ann-file': '/media/cluster_fs/datasets/object_detection/Dice/train.json',
        'train-data-root': '/media/cluster_fs/datasets/object_detection/Dice/export/',
        'val-ann-file': '/media/cluster_fs/datasets/object_detection/Dice/valid.json',
        'val-data-root': '/media/cluster_fs/datasets/object_detection/Dice/export/',
        'test-ann-file': '/media/cluster_fs/datasets/object_detection/Dice/test.json',
        'test-data-root': '/media/cluster_fs/datasets/object_detection/Dice/export/',
    },
    {
        'name': 'MinneApple',
        'train-ann-file': '/media/cluster_fs/datasets/object_detection/MinneApple/detection/train/train_coco.json',
        'train-data-root': '/media/cluster_fs/datasets/object_detection/MinneApple/detection/train/images/',
        'val-ann-file': '/media/cluster_fs/datasets/object_detection/MinneApple/detection/train/val_coco.json',
        'val-data-root': '/media/cluster_fs/datasets/object_detection/MinneApple/detection/train/images/',
        'test-ann-file': '/media/cluster_fs/datasets/object_detection/MinneApple/detection/train/test_coco.json',
        'test-data-root': '/media/cluster_fs/datasets/object_detection/MinneApple/detection/train/images/',
    },
    {
        'name': 'WGISD1',
        'train-ann-file': '/media/cluster_fs/datasets/object_detection/wgisd/train_1_class.json',
        'train-data-root': '/media/cluster_fs/datasets/object_detection/wgisd/original_resolution/',
        'val-ann-file': '/media/cluster_fs/datasets/object_detection/wgisd/test_1_class.json',
        'val-data-root': '/media/cluster_fs/datasets/object_detection/wgisd/original_resolution/',
        'test-ann-file': '/media/cluster_fs/datasets/object_detection/wgisd/test_1_class.json',
        'test-data-root': '/media/cluster_fs/datasets/object_detection/wgisd/original_resolution/',
    },
    {
        'name': 'WGISD5',
        'train-ann-file': '/media/cluster_fs/datasets/object_detection/wgisd/train_5_classes.json',
        'train-data-root': '/media/cluster_fs/datasets/object_detection/wgisd/original_resolution/',
        'val-ann-file': '/media/cluster_fs/datasets/object_detection/wgisd/test_5_classes.json',
        'val-data-root': '/media/cluster_fs/datasets/object_detection/wgisd/original_resolution/',
        'test-ann-file': '/media/cluster_fs/datasets/object_detection/wgisd/test_5_classes.json',
        'test-data-root': '/media/cluster_fs/datasets/object_detection/wgisd/original_resolution/',
    },
    {
        'name': 'PCB-Original',
        'train-ann-file': '/media/cluster_fs/datasets/object_detection/PCB_ORiGINAL-coco/annotations/instances_train.json',
        'train-data-root': '/media/cluster_fs/datasets/object_detection/PCB_ORiGINAL-coco/images/',
        'val-ann-file': '/media/cluster_fs/datasets/object_detection/PCB_ORiGINAL-coco/annotations/instances_val.json',
        'val-data-root': '/media/cluster_fs/datasets/object_detection/PCB_ORiGINAL-coco/images/',
        # 'test-ann-file': '/media/cluster_fs/datasets/object_detection/PCB_ORiGINAL-coco/annotations/instances_test.json', No such file or directory: '/media/cluster_fs/datasets/object_detection/PCB_ORiGINAL-coco/images/.DS_S.jpg'
        # 'test-data-root': '/media/cluster_fs/datasets/object_detection/PCB_ORiGINAL-coco/images/',
    }
]

for dataset in datasets:
    if not dataset['train-ann-file']:
        continue

    run(f"python ../../../../../ote/tools/train.py "
        f"--train-ann-files {dataset['train-ann-file']} "
        f"--train-data-roots {dataset['train-data-root']} "
        f"--val-ann-files {dataset['val-ann-file']} "
        f"--val-data-roots {dataset['val-data-root']} "
        f"--save-checkpoints-to {dataset['name']} "
        , check=True, shell=True)

    for subset in ('train', 'val', 'test'):
        ann_key = f'{subset}-ann-file'
        data_key = f'{subset}-data-root'
        if ann_key not in dataset:
            continue
        run(f"python ../../../../../ote/tools/eval.py "
            f"--load-weights {dataset['name']}/latest.pth "
            f"--test-ann-files {dataset[ann_key]} "
            f"--test-data-roots {dataset[data_key]} "
            f"--save-metrics-to {dataset['name']}/{subset}_metrics.json "
            , check=True, shell=True)

names = []
metrics = []
for dataset in datasets:
    names.append(dataset['name'])
    for subset in ('train', 'val', 'test'):
        try:
            map = load_map(f"{dataset['name']}/{subset}_metrics.json")
            metrics.append(str(map))
        except Exception as e:
            print(dataset['name'], subset, str(e))
            metrics.append('')
    # append empty time, since is not currently estimated
    metrics.append('')

print(','.join(names))
print(','.join(metrics))
