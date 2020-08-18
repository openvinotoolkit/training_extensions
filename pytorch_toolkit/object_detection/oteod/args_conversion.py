import os

from ote import MODEL_TEMPLATE_FILENAME


def convert_ote_to_oteod_train_args(template_folder, args):
    update_config_map = {
        'train_ann_files': 'data.train.dataset.ann_file',
        'train_img_roots': 'data.train.dataset.img_prefix',
        'val_ann_files': 'data.val.ann_file',
        'val_img_roots': 'data.val.img_prefix',
        'resume_from': 'resume_from',
        'load_weights': 'load_from',
        'save_checkpoints_to': 'work_dir',
        'batch_size': 'data.samples_per_gpu',
        'epochs': 'total_epochs',
    }

    update_config = [f'{v}={args[k]}' for k, v in update_config_map.items()]

    oteod_args = {
        'config': os.path.join(template_folder, args['config']),
        'gpu_num': args['training_gpu_num'],
        'out': os.path.join(args['save_checkpoints_to'], MODEL_TEMPLATE_FILENAME),
        'update_config': ' '.join(update_config)
    }

    return oteod_args


def convert_ote_to_oteod_test_args(template_folder, args):
    update_config_map = {
        'test_ann_files': 'data.test.ann_file',
        'test_img_roots': 'data.test.img_prefix',
    }

    update_config = [f'{v}={args[k]}' for k, v in update_config_map.items()]

    oteod_args = {
        'config': os.path.join(template_folder, args['config']),
        'snapshot': args['load_weights'],
        'out': args['save_metrics_to'],
        'update_config': ' '.join(update_config),
        'show_dir': args['save_output_images_to']
    }

    return oteod_args
