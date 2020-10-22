import json
import re
import sys
from datetime import datetime
from pathlib import Path
from pprint import pprint

import torch
from tensorboardX import SummaryWriter
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import WeightedRandomSampler, RandomSampler

from action_recognition.dataset import make_dataset
from action_recognition.logging import TrainingLogger, StreamHandler, TensorboardHandler, CSVHandler
from action_recognition.loss import create_criterion
from action_recognition.model import create_model
from action_recognition.options import parse_arguments
from action_recognition.spatial_transforms import (
    MEAN_STATISTICS, STD_STATISTICS, CenterCrop, Compose, CornerCrop,
    GaussCrop, HorizontalFlip, MultiScaleCrop, Normalize, RandomHorizontalFlip, RandomVerticalFlip,
    Scale, ToTensor, RandomScale, RandomCrop, PadIfNeeded, RandomSharpness, RandomBrightness, RandomContrast
)
from action_recognition.target_transforms import ClassLabel
from action_recognition.temporal_transforms import (
    LoopPadding, TemporalRandomCrop, TemporalStride)
from action_recognition.test import test
from action_recognition.train import train
from action_recognition.utils import (
    TeedStream, json_serialize, load_state,
    create_code_snapshot, mkdir_if_not_exists, print_git_revision)
from action_recognition.validation import validate


def export_onnx(args, model, export_name):
    model = model.module if args.cuda else model
    model.eval()

    if hasattr(model, "export_onnx"):
        model.export_onnx(export_name)
        return

    param = next(model.parameters())
    x = param.new_zeros(1, args.sample_duration, 3, args.sample_size, args.sample_size)

    with torch.no_grad():
        torch.onnx.export(model, (x,), export_name, verbose=True)
    print("Done")


def make_normalization(args):
    if not args.mean_norm and not args.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not args.std_norm:
        norm_method = Normalize(MEAN_STATISTICS[args.mean_dataset], [1, 1, 1])
    else:
        norm_method = Normalize(MEAN_STATISTICS[args.mean_dataset], STD_STATISTICS[args.mean_dataset])
    return norm_method


def setup_dataset(args, train=True, val=True, test=True):
    temporal_stride_size = args.temporal_stride
    sample_duration = args.sample_duration * temporal_stride_size

    normalization = make_normalization(args)

    photometric = []
    if args.photometric:
        photometric = [
            # RandomContrast(),
            RandomSharpness(lower=0.1),
            RandomBrightness(delta=0.75),
            # PhotometricDistort(),
        ]

    train_spatial_transform = [Compose([
        Scale(args.sample_size),
        RandomScale(scale_range=args.scales),
        PadIfNeeded((args.sample_size, args.sample_size)),
        MultiScaleCrop((args.sample_size, args.sample_size), scale_ratios=[1.0])
            if args.crop == 'fixed'
            else RandomCrop(args.sample_size, mode=args.crop),
        *photometric,
        ToTensor(args.norm_value),
        normalization,
    ])]
    if args.hflip:
        train_spatial_transform[0].transforms.insert(1, RandomHorizontalFlip())
    if args.vflip:
        train_spatial_transform[0].transforms.insert(1, RandomVerticalFlip())

    temporal_stride = TemporalStride(temporal_stride_size)
    train_temporal_transform = Compose(
        [temporal_stride, TemporalRandomCrop(sample_duration / temporal_stride_size)])
    train_target_transform = ClassLabel()
    train_data = make_dataset(args, 'training', train_spatial_transform, train_temporal_transform,
                              train_target_transform) if train else None

    # validation data
    val_spatial_transform = [Compose([
        Scale(args.sample_size),
        CenterCrop(args.sample_size),
        ToTensor(args.norm_value),
        normalization
    ])]
    val_temporal_transform = Compose([temporal_stride, LoopPadding(sample_duration / temporal_stride_size)])
    val_target_transform = ClassLabel()
    val_data = make_dataset(args, 'validation', val_spatial_transform, val_temporal_transform,
                            val_target_transform) if val else None

    # test data
    test_spatial_transform = [Compose([
        Scale(int(args.sample_size / args.scale_in_test)),
        CornerCrop(args.sample_size, args.crop_position_in_test),
        ToTensor(args.norm_value), normalization
    ])]

    if args.tta:
        test_spatial_transform = []

        for i in range(5):
            test_spatial_transform.append(Compose([
                Scale(int(args.sample_size / args.scale_in_test)),
                GaussCrop(args.sample_size),
                ToTensor(args.norm_value), normalization
            ]))
            test_spatial_transform.append(Compose([
                Scale(int(args.sample_size / args.scale_in_test)),
                GaussCrop(args.sample_size),
                HorizontalFlip(),
                ToTensor(args.norm_value), normalization
            ]))

    test_temporal_transform = Compose([temporal_stride, LoopPadding(sample_duration / temporal_stride_size)])
    test_target_transform = ClassLabel()

    test_data = make_dataset(args, 'testing', test_spatial_transform, test_temporal_transform,
                             test_target_transform) if test else None

    return train_data, val_data, test_data


def setup_solver(args, parameters):
    if args.optimizer == 'adam':
        optimizer = optim.Adam(parameters, lr=args.learning_rate, weight_decay=args.weight_decay,
                               eps=1e-3 if args.fp16 else 1e-8)
    else:  # args.optimizer == 'sgd'
        optimizer = optim.SGD(parameters, lr=args.learning_rate, weight_decay=args.weight_decay,
                              momentum=0.9, nesterov=args.nesterov)

    if args.scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.lr_patience, threshold=0.01, cooldown=0,
                                                   threshold_mode='abs', mode='max', factor=args.gamma)
    else:  # args.scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    return optimizer, scheduler


def setup_logging(args):
    logger = TrainingLogger()

    # Create handlers
    logger.register_handler("val_batch", StreamHandler(prefix='Val: ', scope='batch'))
    logger.register_handler("val_epoch", StreamHandler(fmt="* {epoch} epochs done:  {values}", scope='epoch',
                                                       display_instant=False))

    logger.register_handler("test_std", StreamHandler(fmt="Testing: [{step}/{total}]\t{values}",
                                                      scope='batch'))
    logger.register_handler("test_end", StreamHandler(fmt="* Testing results: {values}", scope='epoch',
                                                      display_instant=False))
    logger.register_handler("train_epoch", StreamHandler(fmt="* Train epoch {epoch} done:  {values}", scope='epoch'))
    logger.register_handler("train_batch", StreamHandler(prefix='Train: ', scope='batch'))

    logger.register_handler("tb", TensorboardHandler(scope='epoch', summary_writer=args.writer))
    logger.register_handler("tb_global", TensorboardHandler(scope='global', summary_writer=args.writer))

    logger.register_handler("val_csv", CSVHandler(scope='epoch', csv_path=(args.result_path / 'val.csv'),
                                                  index_col='epoch'))
    logger.register_handler("train_csv", CSVHandler(scope='epoch', csv_path=(args.result_path / 'train.csv'),
                                                    index_col='epoch'))

    # Create logged values
    logger.register_value("train/acc", ['train_batch', 'tb_global'], average=True, display_name='clip')
    logger.register_value("train/loss", ['train_batch', 'tb_global'], average=True, display_name='loss')
    logger.register_value("train/kd_loss", ['train_batch', 'tb_global'], average=True, display_name='loss')
    logger.register_value("train/epoch_acc", ['train_epoch', 'tb', 'train_csv'], display_name='clip')
    logger.register_value("train/epoch_loss", ['train_epoch', 'tb', 'train_csv'], display_name='loss')
    logger.register_value_group("lr/.*", ['tb'])

    logger.register_value("time/train_data", ['train_batch'], average=True, display_name='data time')
    logger.register_value("time/train_step", ['train_batch'], average=True, display_name='time')
    logger.register_value("time/train_epoch", ['train_epoch'], display_name='Train epoch time')

    logger.register_value("val/acc", ['val_batch', 'val_epoch', 'tb', 'val_csv'], average=True, display_name='clip')
    logger.register_value("val/video", ['val_batch', 'val_epoch', 'tb', 'val_csv'], average=False, display_name='video')
    logger.register_value("val/loss", ['val_batch', 'tb', 'val_csv'], average=True, display_name='loss')
    logger.register_value("val/generalization_error", ['val_epoch', 'tb', 'val_csv'],
                          display_name='Train Val accuracy gap')

    logger.register_value("time/val_data", ['val_batch'], average=True, display_name='data time')
    logger.register_value("time/val_step", ['val_batch'], average=True, display_name='time')
    logger.register_value("time/val_epoch", ['val_epoch'], average=False, display_name='Validation time')

    logger.register_value("test/acc", ['test_std', 'test_end', 'tb'], average=True, display_name='clip')
    logger.register_value("test/video", ['test_std', 'test_end', 'tb'], average=False, display_name='video')

    return logger


def log_configuration(args):
    print("ARGV:", "-" * 80, sep='\n')
    pprint(sys.argv)
    print()
    print("CONFIG: ", "-" * 80, sep='\n')
    pprint(vars(args))
    print_git_revision()
    print()


def log_training_setup(model, train_data, val_data, optimizer, scheduler):
    print("Model:", model)
    print("Train spatial transforms: ", train_data.spatial_transform)
    print("Train temporal transforms: ", train_data.temporal_transform)
    print("Val spatial transforms: ", val_data.spatial_transform)
    print("Val temporal transforms: ", val_data.temporal_transform)
    print("Optimizer: ", optimizer)
    print("Scheduler: ", scheduler)
    print("-" * 89)


def find_latest_checkpoint(result_path):
    latest_found = -1
    latest_path = None
    for ckpt_path in (result_path / 'checkpoints').iterdir():
        ckpt_name = ckpt_path.name

        match = re.match(r".*_(\d+)\..*", ckpt_name)
        if match and int(match.group(1)) > latest_found:
            latest_found = int(match.group(1))
            latest_path = ckpt_path

    return latest_path


def prepare_result_dir(result_path):
    result_path = Path(result_path)
    mkdir_if_not_exists(result_path)

    # find first directory suffix
    # if directory already ends-up with numeric suffix, use it as result path.
    if not re.match(r'\d+', result_path.parts[-1]):

        files = [str(f.name) for f in result_path.iterdir()]
        i = 1
        while str(i) in files:
            i += 1
        result_path = result_path / str(i)
        result_path.mkdir()
    # create aux dirs
    mkdir_if_not_exists(result_path / 'tb')
    mkdir_if_not_exists(result_path / 'checkpoints')

    return result_path


def configure_paths(args):
    relative_paths = ('video_path', 'annotation_path', 'video_path', 'flow_path', 'resume_path', 'pretrain_path')
    if args.root_path:
        # make paths relative to the args.root_path
        for path in relative_paths:
            arg_path = getattr(args, path, None)
            if arg_path:
                setattr(args, path, args.root_path / arg_path)

    # create directory for storing results (checkpoints, logs, etc.)
    args.result_path = prepare_result_dir(args.result_path)
    # try resume training from latest checkpoint in a result dir
    if args.try_resume and not args.resume_path:
        args.resume_path = find_latest_checkpoint(args.result_path)


def configure_dataset(args):
    dataset = args.dataset
    if dataset in ('hmdb51', 'ucf101'):
        dataset += '_' + str(getattr(args, 'split', 1))

    config_path = Path(__file__).parent / 'datasets' / "{}.json".format(dataset)
    if args.dataset_config:
        config_path = Path(args.dataset_config)
    with config_path.open() as fp:
        dataset_config = json.load(fp)
    if 'flow_path' not in dataset_config:
        args.flow_path = None

    # copy options from dataset config
    for k, v in dataset_config.items():
        if not hasattr(args, k) or not getattr(args, k):
            setattr(args, k, v)


def configure():
    args = parse_arguments()

    configure_dataset(args)
    configure_paths(args)

    with (args.result_path / 'opts.json').open('w') as opt_file:
        json.dump(vars(args), opt_file, default=json_serialize)

    args.tee = TeedStream(args.result_path / "output.log")

    args.time_suffix = datetime.now().strftime("%d%m%H%M")
    tb_path = args.result_path / "tb"

    args.writer = SummaryWriter(tb_path.as_posix())
    args.device = torch.device("cuda" if args.cuda else "cpu")

    create_code_snapshot(Path(__file__).parent, args.result_path / "snapshot.tgz")
    torch.manual_seed(args.manual_seed)

    args.logger = setup_logging(args)
    return args


def main():
    args = configure()
    log_configuration(args)

    model, parameters = create_model(args, args.model, pretrain_path=args.pretrain_path)
    optimizer, scheduler = setup_solver(args, parameters)

    if args.resume_path:
        print('loading checkpoint {}'.format(args.resume_path))
        checkpoint = torch.load(str(args.resume_path), map_location='cpu')
        load_state(model, checkpoint['state_dict'])

        if args.resume_train:
            args.begin_epoch = checkpoint['epoch']
            if not args.train and checkpoint.get('optimizer') is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])

    if args.onnx:
        export_onnx(args, model, args.onnx)
        return

    criterion = create_criterion(args)
    train_data, val_data, test_data = setup_dataset(args, args.train, args.val, args.test)

    if args.train or args.val:
        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.n_threads,
            pin_memory=True,
            drop_last=False
        )

    if args.train:
        if args.weighted_sampling:
            class_weights = getattr(args, 'class_weights', None)
            sampler = WeightedRandomSampler(train_data.get_sample_weights(class_weights), len(train_data))
        else:
            if len(train_data) < args.batch_size:
                sampler = RandomSampler(train_data, replacement=True, num_samples=args.batch_size)
            else:
                sampler = RandomSampler(train_data, replacement=False)

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.n_threads,
            pin_memory=True,
            drop_last=args.sync_bn
        )

        log_training_setup(model, train_data, val_data, optimizer, scheduler)

        train(args, model, train_loader, val_loader, criterion, optimizer, scheduler, args.logger)

    if not args.train and args.val:
        with torch.no_grad():
            validate(args, args.begin_epoch, val_loader, model, criterion, args.logger)

    if args.test:
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.n_threads,
            pin_memory=True
        )

        with torch.no_grad():
            with args.logger.scope():
                test(args, test_loader, model, args.logger)


if __name__ == '__main__':
    main()
