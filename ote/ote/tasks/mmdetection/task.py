# pylint: disable=too-many-branches,too-many-statements,protected-access

import copy
import io
import os
import os.path as osp
import sys
import time
from collections import OrderedDict

import mmcv
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from mmcv.parallel import (MMDataParallel, MMDistributedDataParallel,
                           is_module_wrapper)
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner, OptimizerHook,
                         build_optimizer, get_dist_info, init_dist,
                         load_checkpoint)
from mmcv.runner.checkpoint import load_state_dict, weights_to_cpu
from mmdet import __version__
from mmdet.apis import get_fake_input, set_random_seed, single_gpu_test
from mmdet.core import (DistEvalHook, DistEvalPlusBeforeRunHook, EvalHook,
                        EvalPlusBeforeRunHook, Fp16OptimizerHook)
from mmdet.datasets import build_dataloader
from mmdet.integration.nncf import (CompressionHook, check_nncf_is_enabled,
                                    get_nncf_metadata, wrap_nncf_model)
# from  mmdet.integration.nncf import is_checkpoint_nncf, get_nncf_config_from_meta
from mmdet.models import build_detector
from mmdet.parallel import MMDataCPU
from mmdet.utils import collect_env, get_root_logger
from ote import MMDETECTION_TOOLS
from ote.interfaces.parameters import BaseTaskParameters
from ote.interfaces.task import ITask
from ote.tasks.mmdetection.dataset import CocoDataset2, CocoDataSource
from torch.optim import Optimizer

sys.path.append(f'{MMDETECTION_TOOLS}')
# pylint: disable=wrong-import-position
from export import (add_node_names, export_to_onnx, export_to_openvino,
                    optimize_onnx_graph)


def _save_to_state_dict(module, destination, prefix, keep_vars):
    """Saves module state to `destination` dictionary.

    This method is modified from :meth:`torch.nn.Module._save_to_state_dict`.

    Args:
        module (nn.Module): The module to generate state_dict.
        destination (dict): A dict where state will be stored.
        prefix (str): The prefix for parameters and buffers used in this
            module.
    """
    for name, param in module._parameters.items():
        if param is not None:
            destination[prefix + name] = param if keep_vars else param.detach()
    for name, buf in module._buffers.items():
        # remove check of _non_persistent_buffers_set to allow nn.BatchNorm2d
        if buf is not None:
            destination[prefix + name] = buf if keep_vars else buf.detach()


def get_state_dict(module, destination=None, prefix='', keep_vars=False):
    """Returns a dictionary containing a whole state of the module.

    Both parameters and persistent buffers (e.g. running averages) are
    included. Keys are corresponding parameter and buffer names.

    This method is modified from :meth:`torch.nn.Module.state_dict` to
    recursively check parallel module in case that the model has a complicated
    structure, e.g., nn.Module(nn.Module(DDP)).

    Args:
        module (nn.Module): The module to generate state_dict.
        destination (OrderedDict): Returned dict for the state of the
            module.
        prefix (str): Prefix of the key.
        keep_vars (bool): Whether to keep the variable property of the
            parameters. Default: False.

    Returns:
        dict: A dictionary containing a whole state of the module.
    """
    # recursively check parallel module in case that the model has a
    # complicated structure, e.g., nn.Module(nn.Module(DDP))
    if is_module_wrapper(module):
        module = module.module

    # below is the same as torch.nn.Module.state_dict()
    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()
    destination._metadata[prefix[:-1]] = local_metadata = dict(
        version=module._version)
    _save_to_state_dict(module, destination, prefix, keep_vars)
    for name, child in module._modules.items():
        if child is not None:
            get_state_dict(
                child, destination, prefix + name + '.', keep_vars=keep_vars)
    for hook in module._state_dict_hooks.values():
        hook_result = hook(module, destination, prefix, local_metadata)
        if hook_result is not None:
            destination = hook_result
    return destination


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None,
                   val_dataset=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed) for ds in dataset
    ]

    if cfg.load_from:
        load_checkpoint(model=model, filename=cfg.load_from)

    # put model on gpus
    if torch.cuda.is_available():
        model = model.cuda()

    # nncf model wrapper
    nncf_enable_compression = bool(cfg.get('nncf_config'))
    if nncf_enable_compression:
        compression_ctrl, model = wrap_nncf_model(model, cfg, data_loaders[0], get_fake_input)
    else:
        compression_ctrl = None

    map_location = 'default'
    if torch.cuda.is_available():
        if distributed:
            # put model on gpus
            find_unused_parameters = cfg.get('find_unused_parameters', False)
            # Sets the `find_unused_parameters` parameter in
            # torch.nn.parallel.DistributedDataParallel
            model = MMDistributedDataParallel(
                model,
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model = MMDataParallel(
                model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
    else:
        model = MMDataCPU(model)
        map_location = 'cpu'

    if nncf_enable_compression and distributed:
        compression_ctrl.distributed()

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = EpochBasedRunner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        runner.register_hook(DistSamplerSeedHook())

    #add_logging_on_first_and_last_iter(runner)

    # register eval hooks
    if validate:
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_hook = DistEvalHook if distributed else EvalHook
        if nncf_enable_compression:
            eval_hook = DistEvalPlusBeforeRunHook if distributed else EvalPlusBeforeRunHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if nncf_enable_compression:
        runner.register_hook(CompressionHook(compression_ctrl=compression_ctrl))

    if cfg.resume_from:
        runner.resume(cfg.resume_from, map_location=map_location)

    runner.run(data_loaders, cfg.workflow, cfg.total_epochs, compression_ctrl=compression_ctrl)


def init_dist_cpu(launcher, backend, **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        dist.init_process_group(backend=backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def save_checkpoint_to_buffer(model, optimizer=None, meta=None):
    """Save checkpoint to file.

    The checkpoint will have 3 fields: ``meta``, ``state_dict`` and
    ``optimizer``. By default ``meta`` will contain version and time info.

    Args:
        model (Module): Module whose params are to be saved.
        optimizer (:obj:`Optimizer`, optional): Optimizer to be saved.
        meta (dict, optional): Metadata to be saved in checkpoint.
    """
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError(f'meta must be a dict or None, but got {type(meta)}')
    meta.update(mmcv_version=mmcv.__version__, time=time.asctime())

    if is_module_wrapper(model):
        model = model.module

    if hasattr(model, 'CLASSES') and model.CLASSES is not None:
        # save class name to the meta
        meta.update(CLASSES=model.CLASSES)

    checkpoint = {
        'meta': meta,
        'state_dict': weights_to_cpu(get_state_dict(model))
    }
    # save optimizer state dict in the checkpoint
    if isinstance(optimizer, Optimizer):
        checkpoint['optimizer'] = optimizer.state_dict()
    elif isinstance(optimizer, dict):
        checkpoint['optimizer'] = {}
        for name, optim in optimizer.items():
            checkpoint['optimizer'][name] = optim.state_dict()

    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    return buffer.getvalue()


def load_checkpoint_from_buffer(model,
                                buffer,
                                map_location=None,
                                strict=False,
                                logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = torch.load(io.BytesIO(buffer))
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError('No state_dict found in buffer')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    # load state_dict
    load_state_dict(model, state_dict, strict, logger)

class MMDetectionTask(ITask):

    def init_train_args(self):

        class Args:
            pass

        args = Args()
        args.config = self.env_parameters.config_path
        args.update_config = None
        args.work_dir = self.env_parameters.work_dir
        args.resume_from = None
        args.gpu_ids = None
        args.gpus = None
        args.autoscale_lr = False
        args.launcher = 'none'
        args.seed = None
        args.tensorboard_dir = None
        args.no_validate = False
        args.deterministic = False

        return args


    def __init__(self, parameters: BaseTaskParameters.BaseEnvironmentParameters, load_snapshot=True):
        assert parameters.work_dir

        self.env_parameters = parameters
        self.monitor = None

        self.create_model()
        if load_snapshot:
            load_checkpoint(self.model, self.env_parameters.snapshot_path, map_location='cpu')

    def train(self, train_data_source: CocoDataSource, val_data_source: CocoDataSource,
              parameters: BaseTaskParameters.BaseTrainingParameters=BaseTaskParameters.BaseTrainingParameters()):

        # cfg.train.batch_size = parameters.batch_size
        # cfg.train.lr = parameters.base_learning_rate
        # cfg.train.max_epoch = parameters.max_num_epochs

        args = self.init_train_args()

        cfg = self.config

        cfg.total_epochs = parameters.max_num_epochs

        # if args.resume_from is not None:
        #     cfg.resume_from = args.resume_from
        if args.gpu_ids is not None:
            cfg.gpu_ids = args.gpu_ids
        else:
            cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

        # if args.autoscale_lr:
        #     # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        #     cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

        # init distributed env first, since logger depends on the dist info.
        if args.launcher == 'none':
            distributed = False
        else:
            distributed = True
            if torch.cuda.is_available():
                init_dist(args.launcher, **cfg.dist_params)
            else:
                cfg.dist_params['backend'] = 'gloo'
                init_dist_cpu(args.launcher, **cfg.dist_params)

        # create work_dir
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        # dump config
        print(dir(cfg))
        cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
        # init the logger before other steps
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

        if args.tensorboard_dir is not None:
            hooks = [hook for hook in cfg.log_config.hooks if hook.type == 'TensorboardLoggerHook']
            if hooks:
                hooks[0].log_dir = args.tensorboard_dir
            else:
                logger.warning('Failed to find TensorboardLoggerHook')

        # init the meta dict to record some important information such as
        # environment info and seed, which will be logged
        meta = dict()
        # log env info
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info(f'Environment info:\n{dash_line}{env_info}\n{dash_line}')
        meta['env_info'] = env_info

        # log some basic info
        logger.info(f'Distributed training: {distributed}')
        logger.info(f'Config:\n{cfg.pretty_text}')

        if cfg.get('nncf_config'):
            check_nncf_is_enabled()
            logger.info('NNCF config: {}'.format(cfg.nncf_config))
            meta.update(get_nncf_metadata())

        # set random seeds
        if args.seed is not None:
            logger.info(f'Set random seed to {args.seed}, '
                        f'deterministic: {args.deterministic}')
            set_random_seed(args.seed, deterministic=args.deterministic)
        cfg.seed = args.seed
        meta['seed'] = args.seed

        train_dataset = CocoDataset2(
            data_source = train_data_source,
            pipeline = cfg.data.train.dataset.pipeline,
            classes=cfg.data.train.dataset.classes,
            test_mode=False,
            filter_empty_gt=cfg.data.train.dataset.get('filter_empty_gt', True),
            min_size=cfg.data.train.dataset.get('min_size', None))
        val_dataset = CocoDataset2(
            data_source = val_data_source,
            pipeline = cfg.data.val.pipeline,
            classes=cfg.data.val.classes,
            test_mode=True,
            filter_empty_gt=cfg.data.val.get('filter_empty_gt', True),
            min_size=cfg.data.val.get('min_size', None))

        datasets = [train_dataset]

        dataset_len_per_gpu = sum(len(dataset) for dataset in datasets)
        if distributed:
            dataset_len_per_gpu = dataset_len_per_gpu // get_dist_info()[1]
        assert dataset_len_per_gpu > 0
        if dataset_len_per_gpu < cfg.data.samples_per_gpu:
            cfg.data.samples_per_gpu = dataset_len_per_gpu
            logger.warning(f'Decreased samples_per_gpu to: {cfg.data.samples_per_gpu} '
                        f'because of dataset length: {dataset_len_per_gpu} '
                        f'and gpus number: {get_dist_info()[1]}')

        if len(cfg.workflow) == 2:
            val_dataset = copy.deepcopy(cfg.data.val)
            val_dataset.pipeline = cfg.data.train.pipeline
            datasets.append(val_dataset)
        if cfg.checkpoint_config is not None:
            # save mmdet version, config file content and class names in
            # checkpoints as meta data
            cfg.checkpoint_config.meta = dict(
                mmdet_version=__version__,
                config=cfg.pretty_text,
                CLASSES=cfg.data.train.dataset.classes)
            # also save nncf status in the checkpoint -- it is important,
            # since it is used in wrap_nncf_model for loading NNCF-compressed models
            if cfg.get('nncf_config'):
                nncf_metadata = get_nncf_metadata()
                cfg.checkpoint_config.meta.update(nncf_metadata)
        else:
            # cfg.checkpoint_config is None
            assert not cfg.get('nncf_config'), (
                    "NNCF is enabled, but checkpoint_config is not set -- "
                    "cannot store NNCF metainfo into checkpoints")

        # add an attribute for visualization convenience
        self.model.CLASSES = cfg.data.train.dataset.classes

        train_detector(
            self.model,
            datasets,
            cfg,
            distributed=distributed,
            validate=(not args.no_validate),
            timestamp=timestamp,
            meta=meta,
            val_dataset=val_dataset)

    def test(self,
             dataset: CocoDataSource,
             parameters: BaseTaskParameters.BaseEvaluationParameters) -> (list, dict):
        self.model.eval()

        val_dataset = CocoDataset2(
            data_source = dataset,
            pipeline = self.config.data.test.pipeline,
            classes=self.config.data.test.classes,
            test_mode=True,
            filter_empty_gt=self.config.data.test.get('filter_empty_gt', True),
            min_size=self.config.data.test.get('min_size', None))

        data_loader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=self.config.data.workers_per_gpu,
            dist=False,
            shuffle=False)

        # # nncf model wrapper
        # if is_checkpoint_nncf(args.checkpoint) and not cfg.get('nncf_config'):
        #     # reading NNCF config from checkpoint
        #     nncf_part = get_nncf_config_from_meta(args.checkpoint)
        #     for k, v in nncf_part.items():
        #         cfg[k] = v

        # if cfg.get('nncf_config'):
        #     check_nncf_is_enabled()
        #     if not is_checkpoint_nncf(args.checkpoint):
        #         raise RuntimeError('Trying to make testing with NNCF compression a model snapshot '
        #                            'that was NOT trained with NNCF')
        #     cfg.load_from = args.checkpoint
        #     cfg.resume_from = None
        #     if torch.cuda.is_available():
        #         model = model.cuda()
        #     _, model = wrap_nncf_model(model, cfg, None, get_fake_input)
        #     checkpoint = torch.load(args.checkpoint, map_location=None)
        # else:
        #     checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

        if torch.cuda.is_available():
            model = MMDataParallel(self.model, device_ids=[0])
        else:
            model = MMDataCPU(self.model)

        outputs = single_gpu_test(model, data_loader)

        eval_results = val_dataset.evaluate(outputs, metric='bbox')

        return [], eval_results

    def cancel(self):
        pass

    def get_training_progress(self) -> int:
        return 0

    def compress(self, parameters: BaseTaskParameters.BaseCompressParameters):
        pass

    def export(self, parameters: BaseTaskParameters.BaseExportParameters):
        opset = 11
        alt_ssd_export = False
        args_input_shape = None
        args_input_format = 'BGR'

        parameters.onnx = False

        torch.set_default_tensor_type(torch.FloatTensor)

        self.model.eval()

        if torch.cuda.is_available():
            self.model.cuda()
        device = next(self.model.parameters()).device
        cfg = copy.deepcopy(self.config)
        fake_data = get_fake_input(cfg, device=device)

        # # BEGIN nncf part
        # was_model_compressed = is_checkpoint_nncf(args.checkpoint)
        # cfg_contains_nncf = cfg.get('nncf_config')

        # if cfg_contains_nncf and not was_model_compressed:
        #     raise RuntimeError('Trying to make export with NNCF compression '
        #                     'a model snapshot that was NOT trained with NNCF')

        # if was_model_compressed and not cfg_contains_nncf:
        #     # reading NNCF config from checkpoint
        #     nncf_part = get_nncf_config_from_meta(args.checkpoint)
        #     for k, v in nncf_part.items():
        #         cfg[k] = v

        # if cfg.get('nncf_config'):
        #     alt_ssd_export = getattr(args, 'alt_ssd_export', False)
        #     assert not alt_ssd_export, \
        #             'Export of NNCF-compressed model is incompatible with --alt_ssd_export'
        #     check_nncf_is_enabled()
        #     cfg.load_from = args.checkpoint
        #     cfg.resume_from = None
        #     compression_ctrl, model = wrap_nncf_model(model, cfg, None, get_fake_input)
        #     compression_ctrl.prepare_for_export()
        # # END nncf part

        os.makedirs(osp.abspath(parameters.save_model_to), exist_ok=True)
        onnx_model_path = osp.join(parameters.save_model_to, 'model.onnx')


        assert not (parameters.onnx and parameters.openvino)
        with torch.no_grad():
            export_to_onnx(self.model, fake_data, export_name=onnx_model_path, opset=opset,
                           alt_ssd_export=alt_ssd_export,
                           target='onnx' if parameters.onnx else 'openvino', verbose=False)
            add_node_names(onnx_model_path)
            print(f'ONNX model has been saved to "{onnx_model_path}"')

        optimize_onnx_graph(onnx_model_path)

        with_text = False
        if parameters.openvino and not alt_ssd_export:
            if hasattr(self.model, 'roi_head'):
                if getattr(self.model.roi_head, 'with_text', False):
                    with_text = True

        if parameters.openvino:
            input_shape = list(fake_data['img'][0].shape)
            if args_input_shape:
                input_shape = [1, 3, *args_input_shape]
            export_to_openvino(cfg, onnx_model_path, parameters.save_model_to, input_shape, args_input_format,
                               with_text=with_text)

    def load_model_from_bytes(self, binary_model: bytes):
        load_checkpoint_from_buffer(self.model, binary_model)
        if torch.cuda.is_available():
            self.model.cuda()

    def get_model_bytes(self) -> bytes:
        return save_checkpoint_to_buffer(self.model)

    def create_model(self):
        self.config = mmcv.Config.fromfile(self.env_parameters.config_path)
        self.config.work_dir = self.env_parameters.work_dir
        self.model = build_detector(self.config.model,
                                    train_cfg=self.config.train_cfg,
                                    test_cfg=self.config.test_cfg)
