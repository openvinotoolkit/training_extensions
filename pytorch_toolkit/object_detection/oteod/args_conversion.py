import os

from ote import MODEL_TEMPLATE_FILENAME


def convert_ote_to_oteod_train_args(template_folder, args):
    update_config = []
    if args.train_ann_files:
        update_config.append(f'data.train.dataset.ann_file={args.train_ann_files}')
    if args.train_img_roots:
        update_config.append(f'data.train.dataset.img_prefix={args.train_img_roots}')
    if args.val_ann_files:
        update_config.append(f'data.val.ann_file={args.val_ann_files}')
    if args.val_img_roots:
        update_config.append(f'data.val.img_prefix={args.val_img_roots}')
    if args.resume_from:
        update_config.append(f'resume_from={args.resume_from}')
    if args.load_weights:
        update_config.append(f'load_from={args.load_weights}')
    if args.save_checkpoints_to:
        update_config.append(f'work_dir={args.save_checkpoints_to}')

    update_config.append(f'data.samples_per_gpu={args.batch_size}')
    update_config.append(f'total_epochs={args.epochs}')

    oteod_args = {
        'config': os.path.join(template_folder, args.config),
        'gpu_num': args.training_gpu_num,
        'out': os.path.join(args.save_checkpoints_to, MODEL_TEMPLATE_FILENAME),
        'update_config': ' '.join(update_config) if update_config else ''
    }

    return oteod_args


def convert_ote_to_oteod_test_args(template_folder, args):
    update_config = []
    if args.test_ann_files:
        update_config.append(f'data.test.ann_file={args.test_ann_files}')
    if args.test_img_roots:
        update_config.append(f'data.test.img_prefix={args.test_img_roots}')

    oteod_args = {
        'config': os.path.join(template_folder, args.config),
        'snapshot': args.load_weights,
        'out': args.save_metrics_to,
        'update_config': ' '.join(update_config) if update_config else '',
        'show_dir': args.save_output_images_to
    }

    return oteod_args
