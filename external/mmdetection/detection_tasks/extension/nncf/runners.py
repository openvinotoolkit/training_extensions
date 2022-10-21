# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import time

from mmcv.runner import EpochBasedRunner
from mmcv.runner import RUNNERS
from mmcv.runner.utils import get_host_info

from .utils import check_nncf_is_enabled


@RUNNERS.register_module()
class AccuracyAwareRunner(EpochBasedRunner):
    """
    An mmdet training runner to be used with NNCF-based accuracy-aware training.
    Inherited from the standard EpochBasedRunner with the overridden "run" method.
    This runner does not use the "workflow" and "max_epochs" parameters that are
    used by the EpochBasedRunner since the training is controlled by NNCF's
    AdaptiveCompressionTrainingLoop that does the scheduling of the compression-aware
    training loop using the parameters specified in the "accuracy_aware_training".
    """

    def __init__(
            self, *args, target_metric_name,
            compression_ctrl=None, nncf_config=None, optimizer_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_metric_name = target_metric_name
        self.nncf_config = nncf_config
        self.optimizer_config = optimizer_config
        self.compression_ctrl = compression_ctrl

    def run(self, data_loaders, *args, **kwargs):

        check_nncf_is_enabled()
        from nncf.common.accuracy_aware_training import create_accuracy_aware_training_loop
        assert isinstance(data_loaders, list)

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.warning('Note that the workflow and max_epochs parameters '
                            'are not used in NNCF-based accuracy-aware training')

        acc_aware_training_loop = create_accuracy_aware_training_loop(self.nncf_config,
                                                                      self.compression_ctrl,
                                                                      verbose=False)
        # taking only the first data loader for NNCF training
        self.train_data_loader = data_loaders[0]
        # Maximum possible number of iterations, needs for progress tracking
        self._max_epochs = acc_aware_training_loop.runner.maximal_total_epochs
        self._max_iters = self._max_epochs * len(self.train_data_loader)

        self.call_hook('before_run')
        model = acc_aware_training_loop.run(self.model,
                                            train_epoch_fn=self.train_fn,
                                            validate_fn=self.validation_fn,
                                            configure_optimizers_fn=self.configure_optimizers_fn)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')
        return model

    def train_fn(self, *args, **kwargs):
        """
        Train the model for a single epoch.
        This method is used in NNCF-based accuracy-aware training.
        """
        print('----------------> train')
        self.train(self.train_data_loader)
        print('----------------> train end')

    def validation_fn(self, *args, **kwargs):
        """
        Return the target metric value on the validation dataset.
        Evaluation is assumed to be already done at this point since EvalHook was called.
        This method is used in NNCF-based accuracy-aware training.
        """
        print('----------------> val')
        ## Get metric from runner's attributes that set in EvalHook.evaluate() function
        #metric = getattr(self, self.target_metric_name, None)
        #if metric is None:
        #    raise RuntimeError(f'Could not find the {self.target_metric_name} key')
        #return metric
        return 1.0

    def configure_optimizers_fn(self):
        return self.optimizer, None

