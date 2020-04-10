"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from yacs.config import CfgNode as CN


def get_default_config():
    cfg = CN()

    # model
    cfg.model = CN()
    cfg.model.name = 'osnet_x1.0'
    cfg.model.pretrained = True  # Automatically load pretrained model weights if available
    cfg.model.load_weights = ''  # Path to model weights
    cfg.model.resume = ''  # Path to checkpoint for resume training
    cfg.model.dropout = CN()
    cfg.model.dropout.p = 0.0
    cfg.model.dropout.mu = 0.1
    cfg.model.dropout.sigma = 0.03
    cfg.model.dropout.dist = 'none'
    cfg.model.feature_dim = 512
    cfg.model.pooling_type = 'avg'
    cfg.model.IN_first = False
    cfg.model.extra_blocks = False
    cfg.model.openvino = CN()
    cfg.model.openvino.name = ''  # Path to *.xml
    cfg.model.openvino.cpu_extension = ''
    cfg.model.fpn = CN()
    cfg.model.fpn.enable = True
    cfg.model.fpn.dim = 256
    cfg.model.fpn.process = 'concatenation'

    # data
    cfg.data = CN()
    cfg.data.type = 'image'
    cfg.data.root = 'reid-data'
    cfg.data.sources = ['market1501']
    cfg.data.targets = ['market1501']
    cfg.data.workers = 4  # Number of data loading workers
    cfg.data.split_id = 0  # Split index
    cfg.data.height = 256  # Image height
    cfg.data.width = 128  # Image width
    cfg.data.combineall = False  # Combine train, query and gallery for training
    cfg.data.norm_mean = [0.485, 0.456, 0.406]  # Default is imagenet mean
    cfg.data.norm_std = [0.229, 0.224, 0.225]  # Default is imagenet std
    cfg.data.save_dir = 'log'  # Path to save log

    # specific datasets
    cfg.market1501 = CN()
    cfg.market1501.use_500k_distractors = False  # Add 500k distractors to the gallery set for market1501
    cfg.cuhk03 = CN()
    cfg.cuhk03.labeled_images = False  # Use labeled images, if False, use detected images
    cfg.cuhk03.classic_split = False  # Use classic split by Li et al. CVPR14
    cfg.cuhk03.use_metric_cuhk03 = False  # Use cuhk03's metric for evaluation

    # sampler
    cfg.sampler = CN()
    cfg.sampler.train_sampler = 'RandomSampler'
    cfg.sampler.num_instances = 4  # Number of instances per identity for RandomIdentitySampler

    # video reid setting
    cfg.video = CN()
    cfg.video.seq_len = 15  # Number of images to sample in a tracklet
    cfg.video.sample_method = 'evenly'  # How to sample images from a tracklet
    cfg.video.pooling_method = 'avg'  # How to pool features over a tracklet

    # train
    cfg.train = CN()
    cfg.train.optim = 'adam'
    cfg.train.lr = 0.0003
    cfg.train.weight_decay = 5e-4
    cfg.train.max_epoch = 60
    cfg.train.start_epoch = 0
    cfg.train.batch_size = 32
    cfg.train.fixbase_epoch = 0  # Number of epochs to fix base layers
    cfg.train.open_layers = ['classifier']  # Layers for training while keeping others frozen
    cfg.train.staged_lr = False  # Set different lr to different layers
    cfg.train.new_layers = ['classifier']  # Newly added layers with default lr
    cfg.train.base_lr_mult = 0.1  # Learning rate multiplier for base layers
    cfg.train.lr_scheduler = 'single_step'
    cfg.train.stepsize = [20]  # Stepsize to decay learning rate
    cfg.train.gamma = 0.1  # Learning rate decay multiplier
    cfg.train.print_freq = 20  # Print frequency
    cfg.train.seed = 1  # Random seed
    cfg.train.warmup = 1   # After fixbase_epoch
    cfg.train.warmup_factor_base = 0.1

    # optimizer
    cfg.sgd = CN()
    cfg.sgd.momentum = 0.9  # Momentum factor for sgd and rmsprop
    cfg.sgd.dampening = 0.  # Dampening for momentum
    cfg.sgd.nesterov = False  # Nesterov momentum
    cfg.rmsprop = CN()
    cfg.rmsprop.alpha = 0.99  # Smoothing constant
    cfg.adam = CN()
    cfg.adam.beta1 = 0.9  # Exponential decay rate for first moment
    cfg.adam.beta2 = 0.999  # Exponential decay rate for second moment

    # loss
    cfg.loss = CN()
    cfg.loss.name = 'softmax'
    cfg.loss.softmax = CN()
    cfg.loss.softmax.label_smooth = True  # Use label smoothing regularizer
    cfg.loss.softmax.conf_pen = 0.0
    cfg.loss.softmax.pr_product = False
    cfg.loss.softmax.m = 0.35
    cfg.loss.softmax.s = 30
    cfg.loss.triplet = CN()
    cfg.loss.triplet.margin = 0.3  # Distance margin
    cfg.loss.triplet.weight_t = 1.0  # Weight to balance hard triplet loss
    cfg.loss.triplet.weight_x = 0.  # Weight to balance cross entropy loss

    # metric_losses
    cfg.metric_losses = CN()
    cfg.metric_losses.enable = False
    cfg.metric_losses.local_push_weight = 0.0

    # regularizers
    cfg.reg = CN()
    cfg.reg.ow = False
    cfg.reg.ow_beta = 1e-3
    cfg.reg.of = False
    cfg.reg.of_beta = 1e-6
    cfg.reg.of_start_epoch = 23

    # test
    cfg.test = CN()
    cfg.test.batch_size = 100
    cfg.test.dist_metric = 'cosine'  # Distance metric, ['euclidean', 'cosine']
    cfg.test.normalize_feature = False  # Normalize feature vectors before computing distance
    cfg.test.ranks = [1, 5, 10, 20]  # Cmc ranks
    cfg.test.evaluate = False  # Test only
    cfg.test.eval_freq = -1  # Evaluation frequency (-1 means to only test after training)
    cfg.test.start_eval = 0  # Start to evaluate after a specific epoch
    cfg.test.rerank = False  # Use person re-ranking
    cfg.test.visrank = False  # Visualize ranked results (only available when cfg.test.evaluate=True)
    cfg.test.visrank_topk = 10  # Top-k ranks to visualize
    cfg.test.visactmap = False  # Visualize CNN activation maps
    cfg.test.apply_masks = False
    cfg.test.use_flip = False

    # Augmentations
    cfg.data.transforms = CN()

    cfg.data.transforms.random_flip = CN()
    cfg.data.transforms.random_flip.enable = True
    cfg.data.transforms.random_flip.p = 0.5

    cfg.data.transforms.random_crop = CN()
    cfg.data.transforms.random_crop.enable = False
    cfg.data.transforms.random_crop.p = 0.5

    cfg.data.transforms.random_gray_scale = CN()
    cfg.data.transforms.random_gray_scale.enable = False
    cfg.data.transforms.random_gray_scale.p = 0.5

    cfg.data.transforms.random_padding = CN()
    cfg.data.transforms.random_padding.enable = False
    cfg.data.transforms.random_padding.p = 0.5
    cfg.data.transforms.random_padding.padding = (0, 10)

    cfg.data.transforms.random_perspective = CN()
    cfg.data.transforms.random_perspective.enable = False
    cfg.data.transforms.random_perspective.p = 0.5
    cfg.data.transforms.random_perspective.distortion_scale = 0.5

    cfg.data.transforms.color_jitter = CN()
    cfg.data.transforms.color_jitter.enable = False
    cfg.data.transforms.color_jitter.p = 0.5
    cfg.data.transforms.color_jitter.brightness = 0.2
    cfg.data.transforms.color_jitter.contrast = 0.15
    cfg.data.transforms.color_jitter.saturation = 0.0
    cfg.data.transforms.color_jitter.hue = 0.0

    cfg.data.transforms.random_erase = CN()
    cfg.data.transforms.random_erase.enable = False
    cfg.data.transforms.random_erase.p = 0.5
    cfg.data.transforms.random_erase.sl = 0.2
    cfg.data.transforms.random_erase.sh = 0.4
    cfg.data.transforms.random_erase.r1 = 0.3
    cfg.data.transforms.random_erase.mean = (0.4914, 0.4822, 0.4465)

    cfg.data.transforms.random_rotate = CN()
    cfg.data.transforms.random_rotate.enable = False
    cfg.data.transforms.random_rotate.p = 0.5
    cfg.data.transforms.random_rotate.angle = (-5, 5)

    cfg.data.transforms.masks = CN()
    cfg.data.transforms.masks.enable = False
    cfg.data.transforms.masks.p = 0.33
    cfg.data.transforms.masks.random_color = True
    cfg.data.transforms.masks.noise = True

    cfg.data.transforms.random_figures = CN()
    cfg.data.transforms.random_figures.enable = False
    cfg.data.transforms.random_figures.p = 0.5
    cfg.data.transforms.random_figures.random_color = True
    cfg.data.transforms.random_figures.always_single_figure = False
    cfg.data.transforms.random_figures.thicknesses = (1, 6)
    cfg.data.transforms.random_figures.circle_radiuses = (5, 64)
    cfg.data.transforms.random_figures.figure_prob = 0.5

    cfg.data.transforms.random_patch = CN()
    cfg.data.transforms.random_patch.enable = False
    cfg.data.transforms.random_patch.p = 0.5
    cfg.data.transforms.random_patch.pool_capacity = 50000
    cfg.data.transforms.random_patch.min_sample_size = 100
    cfg.data.transforms.random_patch.patch_min_area = 0.01
    cfg.data.transforms.random_patch.patch_max_area = 0.5
    cfg.data.transforms.random_patch.patch_min_ratio = 0.1
    cfg.data.transforms.random_patch.prob_rotate = 0.5
    cfg.data.transforms.random_patch.prob_flip_leftright = 0.5

    cfg.data.transforms.random_grid = CN()
    cfg.data.transforms.random_grid.enable = False
    cfg.data.transforms.random_grid.p = 0.33
    cfg.data.transforms.random_grid.color = (-1, -1, -1)
    cfg.data.transforms.random_grid.grid_size = (24, 64)
    cfg.data.transforms.random_grid.thickness = (1, 1)
    cfg.data.transforms.random_grid.angle = (0, 180)

    cfg.data.transforms.batch_transform = CN()
    cfg.data.transforms.batch_transform.enable = False
    cfg.data.transforms.batch_transform.type = 'Pairing'
    cfg.data.transforms.batch_transform.alpha = 1.
    cfg.data.transforms.batch_transform.anchor_bias = 0.8

    return cfg


def imagedata_kwargs(cfg):
    return {
        'root': cfg.data.root,
        'sources': cfg.data.sources,
        'targets': cfg.data.targets,
        'height': cfg.data.height,
        'width': cfg.data.width,
        'transforms': cfg.data.transforms,
        'norm_mean': cfg.data.norm_mean,
        'norm_std': cfg.data.norm_std,
        'use_gpu': cfg.use_gpu,
        'split_id': cfg.data.split_id,
        'combineall': cfg.data.combineall,
        'batch_size_train': cfg.train.batch_size,
        'batch_size_test': cfg.test.batch_size,
        'workers': cfg.data.workers,
        'num_instances': cfg.sampler.num_instances,
        'train_sampler': cfg.sampler.train_sampler,
        # image
        'cuhk03_labeled': cfg.cuhk03.labeled_images,
        'cuhk03_classic_split': cfg.cuhk03.classic_split,
        'market1501_500k': cfg.market1501.use_500k_distractors,
        'apply_masks_to_test': cfg.test.apply_masks,
    }


def key_points(cfg):
    return {
        'use': cfg.data.use_keypoints,
        'root_path': cfg.data.keypoints
    }


def videodata_kwargs(cfg):
    return {
        'root': cfg.data.root,
        'sources': cfg.data.sources,
        'targets': cfg.data.targets,
        'height': cfg.data.height,
        'width': cfg.data.width,
        'transforms': cfg.data.transforms,
        'norm_mean': cfg.data.norm_mean,
        'norm_std': cfg.data.norm_std,
        'use_gpu': cfg.use_gpu,
        'split_id': cfg.data.split_id,
        'combineall': cfg.data.combineall,
        'batch_size_train': cfg.train.batch_size,
        'batch_size_test': cfg.test.batch_size,
        'workers': cfg.data.workers,
        'num_instances': cfg.sampler.num_instances,
        'train_sampler': cfg.sampler.train_sampler,
        # video
        'seq_len': cfg.video.seq_len,
        'sample_method': cfg.video.sample_method
    }


def optimizer_kwargs(cfg):
    return {
        'optim': cfg.train.optim,
        'lr': cfg.train.lr,
        'weight_decay': cfg.train.weight_decay,
        'momentum': cfg.sgd.momentum,
        'sgd_dampening': cfg.sgd.dampening,
        'sgd_nesterov': cfg.sgd.nesterov,
        'rmsprop_alpha': cfg.rmsprop.alpha,
        'adam_beta1': cfg.adam.beta1,
        'adam_beta2': cfg.adam.beta2,
        'staged_lr': cfg.train.staged_lr,
        'new_layers': cfg.train.new_layers,
        'base_lr_mult': cfg.train.base_lr_mult
    }


def lr_scheduler_kwargs(cfg):
    return {
        'lr_scheduler': cfg.train.lr_scheduler,
        'stepsize': cfg.train.stepsize,
        'gamma': cfg.train.gamma,
        'max_epoch': cfg.train.max_epoch,
        'warmup': cfg.train.warmup,
        'frozen': cfg.train.fixbase_epoch,
        'warmup_factor_base': cfg.train.warmup_factor_base
    }


def engine_run_kwargs(cfg):
    return {
        'save_dir': cfg.data.save_dir,
        'max_epoch': cfg.train.max_epoch,
        'start_epoch': cfg.train.start_epoch,
        'fixbase_epoch': cfg.train.fixbase_epoch,
        'open_layers': cfg.train.open_layers,
        'start_eval': cfg.test.start_eval,
        'eval_freq': cfg.test.eval_freq,
        'test_only': cfg.test.evaluate,
        'print_freq': cfg.train.print_freq,
        'dist_metric': cfg.test.dist_metric,
        'normalize_feature': cfg.test.normalize_feature,
        'visrank': cfg.test.visrank,
        'visrank_topk': cfg.test.visrank_topk,
        'use_metric_cuhk03': cfg.cuhk03.use_metric_cuhk03,
        'ranks': cfg.test.ranks,
        'rerank': cfg.test.rerank,
        'visactmap': cfg.test.visactmap,
    }


def transforms(cfg):
    return cfg.data.transforms


def augmentation_kwargs(cfg):
    return {
        'random_flip': cfg.data.transforms.random_flip,
        'random_crop': cfg.data.transforms.random_crop,
        'random_gray_scale': cfg.data.transforms.random_gray_scale,
        'random_padding': cfg.data.transforms.random_padding,
        'random_perspective': cfg.data.transforms.random_perspective,
        'color_jitter': cfg.data.transforms.color_jitter,
        'random_erase': cfg.data.transforms.random_erase,
        'random_rotate': cfg.data.transforms.random_rotate,
        'masks': cfg.data.transforms.masks,
        'random_figures': cfg.data.transforms.random_figures,
        'random_grid': cfg.data.transforms.random_grid
    }
