# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

"""This implementation copy from mmengine BaseModule."""

from __future__ import annotations

import copy
import logging
from abc import ABCMeta
from collections import defaultdict
from logging import FileHandler
from typing import Iterable

from torch import nn

from otx.algo.utils.weight_init import PretrainedInit, initialize, update_init_info

logger = logging.getLogger()


class BaseModule(nn.Module, metaclass=ABCMeta):
    """Base module for all modules in openmmlab.

    ``BaseModule`` is a wrapper of
    ``torch.nn.Module`` with additional functionality of parameter
    initialization. Compared with ``torch.nn.Module``, ``BaseModule`` mainly
    adds three attributes.

    - ``init_cfg``: the config to control the initialization.
    - ``init_weights``: The function of parameter initialization and recording
      initialization information.
    - ``_params_init_info``: Used to track the parameter initialization
      information. This attribute only exists during executing the
      ``init_weights``.

    Note:
        :obj:`PretrainedInit` has a higher priority than any other
        initializer. The loaded pretrained weights will overwrite
        the previous initialized weights.

    Args:
        init_cfg (dict or List[dict], optional): Initialization config dict.
    """

    def __init__(self, init_cfg: dict | list[dict] | None = None):
        """Initialize BaseModule, inherited from `torch.nn.Module`."""
        # NOTE init_cfg can be defined in different levels, but init_cfg
        # in low levels has a higher priority.

        super().__init__()
        # define default value of init_cfg instead of hard code
        # in init_weights() function
        self._is_init = False

        self.init_cfg = copy.deepcopy(init_cfg)

        # Backward compatibility in derived classes
        # if pretrained is not None:
        #     warnings.warn('DeprecationWarning: pretrained is a deprecated \
        #         key, please consider using init_cfg')
        #     self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

    @property
    def is_init(self) -> bool:
        """Check if the module has been initialized."""
        return self._is_init

    @is_init.setter
    def is_init(self, value: bool) -> None:
        self._is_init = value

    def init_weights(self) -> None:
        """Initialize the weights."""
        is_top_level_module = False
        # check if it is top-level module
        if not hasattr(self, "_params_init_info"):
            # The `_params_init_info` is used to record the initialization
            # information of the parameters
            # the key should be the obj:`nn.Parameter` of model and the value
            # should be a dict containing
            # - init_info (str): The string that describes the initialization.
            # - tmp_mean_value (FloatTensor): The mean of the parameter,
            #       which indicates whether the parameter has been modified.
            # this attribute would be deleted after all parameters
            # is initialized.
            self._params_init_info = defaultdict(dict)  # type: ignore[var-annotated]
            is_top_level_module = True

            # Initialize the `_params_init_info`,
            # When detecting the `tmp_mean_value` of
            # the corresponding parameter is changed, update related
            # initialization information
            for _, param in self.named_parameters():
                self._params_init_info[param]["init_info"] = (
                    f"The value is the same before and "
                    f"after calling `init_weights` "
                    f"of {self.__class__.__name__} "
                )
                self._params_init_info[param]["tmp_mean_value"] = param.data.mean().cpu()

            # pass `params_init_info` to all submodules
            # All submodules share the same `params_init_info`,
            # so it will be updated when parameters are
            # modified at any level of the model.
            for sub_module in self.modules():
                sub_module._params_init_info = self._params_init_info  # noqa: SLF001

        module_name = self.__class__.__name__
        if not self._is_init:
            if self.init_cfg:
                logger.debug(f"initialize {module_name} with init_cfg {self.init_cfg}", stacklevel=1)

                init_cfgs = self.init_cfg
                if isinstance(self.init_cfg, dict):
                    init_cfgs = [self.init_cfg]

                # PretrainedInit has higher priority than any other init_cfg.
                # Therefore we initialize `pretrained_cfg` last to overwrite
                # the previous initialized weights.
                # See details in https://github.com/open-mmlab/mmengine/issues/691 # E501
                other_cfgs = []
                pretrained_cfg = []
                for init_cfg in init_cfgs:
                    assert isinstance(init_cfg, dict)  # noqa: S101
                    if init_cfg["type"] == "Pretrained" or init_cfg["type"] is PretrainedInit:
                        pretrained_cfg.append(init_cfg)
                    else:
                        other_cfgs.append(init_cfg)

                initialize(self, other_cfgs)

            for m in self.children():
                # if is_model_wrapper(m) and not hasattr(m, 'init_weights'):
                #     m = m.module
                if hasattr(m, "init_weights") and not getattr(m, "is_init", False):
                    m.init_weights()
                    # users may overload the `init_weights`
                    update_init_info(
                        m,
                        init_info=f"Initialized by user-defined `init_weights` in {m.__class__.__name__} ",
                    )
            if self.init_cfg and pretrained_cfg:
                initialize(self, pretrained_cfg)
            self._is_init = True
        else:
            logger.warning(f"init_weights of {self.__class__.__name__} has been called more than once.")

        if is_top_level_module:
            self._dump_init_info()

            for sub_module in self.modules():
                del sub_module._params_init_info  # noqa: SLF001

    def _dump_init_info(self) -> None:
        """Dump the initialization information to a file named `initialization.log.json` in workdir."""
        with_file_handler = False
        # dump the information to the logger file if there is a `FileHandler`
        for handler in logger.handlers:
            if isinstance(handler, FileHandler) and handler.stream is not None:
                handler.stream.write("Name of parameter - Initialization information\n")
                for name, param in self.named_parameters():
                    handler.stream.write(
                        f'\n{name} - {param.shape}: ' f"\n{self._params_init_info[param]['init_info']} \n",  # noqa: ISC001
                    )
                handler.stream.flush()
                with_file_handler = True
        if not with_file_handler:
            for name, param in self.named_parameters():
                logger.info(f'\n{name} - {param.shape}: ' f"\n{self._params_init_info[param]['init_info']} \n ")  # noqa: ISC001

    def __repr__(self):
        s = super().__repr__()
        if self.init_cfg:
            s += f"\ninit_cfg={self.init_cfg}"
        return s


class Sequential(BaseModule, nn.Sequential):
    """Sequential module in openmmlab.

    # TODO (sungchul): remove it

    Ensures that all modules in ``Sequential`` have a different initialization
    strategy than the outer model

    Args:
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, *args, init_cfg: dict | None = None):
        BaseModule.__init__(self, init_cfg)
        nn.Sequential.__init__(self, *args)


class ModuleList(BaseModule, nn.ModuleList):
    """ModuleList in openmmlab.

    # TODO (sungchul): remove it

    Ensures that all modules in ``ModuleList`` have a different initialization
    strategy than the outer model

    Args:
        modules (iterable, optional): An iterable of modules to add.
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, modules: Iterable | None = None, init_cfg: dict | None = None):
        BaseModule.__init__(self, init_cfg)
        nn.ModuleList.__init__(self, modules)


class ModuleDict(BaseModule, nn.ModuleDict):
    """ModuleDict in openmmlab.

    # TODO (sungchul): remove it

    Ensures that all modules in ``ModuleDict`` have a different initialization
    strategy than the outer model

    Args:
        modules (dict, optional): A mapping (dictionary) of (string: module)
            or an iterable of key-value pairs of type (string, module).
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, modules: dict | None = None, init_cfg: dict | None = None):
        BaseModule.__init__(self, init_cfg)
        nn.ModuleDict.__init__(self, modules)
