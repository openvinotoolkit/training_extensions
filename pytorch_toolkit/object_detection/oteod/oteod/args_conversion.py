import os

from ote import MODEL_TEMPLATE_FILENAME


def convert_ote_to_oteod_train_args(template_folder, args):
    update_config_map = {
        'train_ann_files': 'data.train.dataset.ann_file',
        'train_data_roots': 'data.train.dataset.img_prefix',
        'val_ann_files': 'data.val.ann_file',
        'val_data_roots': 'data.val.img_prefix',
        'resume_from': 'resume_from',
        'load_weights': 'load_from',
        'save_checkpoints_to': 'work_dir',
        'batch_size': 'data.samples_per_gpu',
        'base_learning_rate': 'optimizer.lr',
        'epochs': 'total_epochs',
    }

    update_config = {v: args[k] for k, v in update_config_map.items()}

    if 'classes' in args and args['classes']:
        classes = '[' + ','.join(f'"{x}"' for x in args['classes'].split(',')) + ']'
        update_config['data.train.dataset.classes'] = classes
        update_config['data.val.classes'] = classes
        update_config['model.bbox_head.num_classes'] = len(args['classes'].split(','))

    oteod_args = {
        'config': os.path.join(template_folder, args['config']),
        'gpu_num': args['gpu_num'],
        'out': os.path.join(args['save_checkpoints_to'], MODEL_TEMPLATE_FILENAME),
        'update_config': update_config,
        'tensorboard_dir': args['tensorboard_dir']
    }

    return oteod_args


def convert_ote_to_oteod_test_args(template_folder, args):
    update_config_map = {
        'test_ann_files': 'data.test.ann_file',
        'test_data_roots': 'data.test.img_prefix',
    }

    update_config = {v: args[k] for k, v in update_config_map.items()}

    if 'classes' in args and args['classes']:
        classes = '[' + ','.join(f'"{x}"' for x in args['classes'].split(',')) + ']'
        update_config['data.test.classes'] = classes
        update_config['model.bbox_head.num_classes'] = len(args['classes'].split(','))

    oteod_args = {
        'config': os.path.join(template_folder, args['config']),
        'snapshot': args['load_weights'],
        'out': args['save_metrics_to'],
        'update_config': update_config,
        'show_dir': args['save_output_to'],
    }

    return oteod_args
