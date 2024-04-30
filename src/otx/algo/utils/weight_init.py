# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

"""This implementation replaces the functionality of mmengine.model.weight_init."""
from __future__ import annotations

import copy
import logging
import math
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

logger = logging.getLogger()


def update_init_info(module: nn.Module, init_info: str) -> None:
    """Update the `_params_init_info` in the module if the value of parameters are changed.

    Args:
        module (obj:`nn.Module`): The module of PyTorch with a user-defined
            attribute `_params_init_info` which records the initialization
            information.
        init_info (str): The string that describes the initialization.
    """
    assert hasattr(  # noqa: S101
        module,
        "_params_init_info",
    ), f"Can not find `_params_init_info` in {module}"
    for name, param in module.named_parameters():
        assert param in module._params_init_info, (  # noqa: S101, SLF001
            f"Find a new :obj:`Parameter` "
            f"named `{name}` during executing the "
            f"`init_weights` of "
            f"`{module.__class__.__name__}`. "
            f"Please do not add or "
            f"replace parameters during executing "
            f"the `init_weights`. "
        )

        # The parameter has been changed during executing the
        # `init_weights` of module
        mean_value = param.data.mean().cpu()
        if module._params_init_info[param]["tmp_mean_value"] != mean_value:  # noqa: SLF001
            module._params_init_info[param]["init_info"] = init_info  # noqa: SLF001
            module._params_init_info[param]["tmp_mean_value"] = mean_value  # noqa: SLF001


def constant_init(module: nn.Module, val: float, bias: float = 0) -> None:
    """Initialize the weights and biases of a module with constant values.

    Copied from mmengine.model.weight_init.constant_init

    Args:
        module (nn.Module): The module to initialize.
        val (float): The constant value to initialize the weights with.
        bias (float, optional): The constant value to initialize the biases with. Defaults to 0.
    """
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module: nn.Module, gain: int | float = 1, bias: int | float = 0, distribution: str = "normal") -> None:
    """Initialize the weights and biases of a module using Xavier initialization.

    Args:
        module (nn.Module): The module to initialize.
        gain (int | float): The scaling factor for the weights. Default is 1.
        bias (int | float): The bias value. Default is 0.
        distribution (str): The distribution to use for weight initialization. Can be 'uniform' or 'normal'.
            Default is 'normal'.
    """
    assert distribution in ["uniform", "normal"]  # noqa: S101
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module: nn.Module, mean: float = 0, std: float = 1, bias: float = 0) -> None:
    """Initialize the weights and biases of a module using a normal distribution.

    Copied from mmengine.model.weight_init.normal_init

    Args:
        module (nn.Module): The module to initialize.
        mean (float): The mean of the normal distribution. Default is 0.
        std (float): The standard deviation of the normal distribution. Default is 1.
        bias (float): The bias value. Default is 0.
    """
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def trunc_normal_init(
    module: nn.Module,
    mean: float = 0,
    std: float = 1,
    a: float = -2,
    b: float = 2,
    bias: float = 0,
) -> None:
    """Initialize the weights and biases of a module using a truncated normal distribution.

    Args:
        module (nn.Module): The module to initialize.
        mean (float): The mean of the truncated normal distribution. Default is 0.
        std (float): The standard deviation of the truncated normal distribution. Default is 1.
        a (float): The lower bound of the truncated normal distribution. Default is -2.
        b (float): The upper bound of the truncated normal distribution. Default is 2.
        bias (float): The bias value. Default is 0.
    """
    if hasattr(module, "weight") and module.weight is not None:
        trunc_normal_(module.weight, mean, std, a, b)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def uniform_init(module: nn.Module, a: int | float = 0, b: int | float = 1, bias: int | float = 0) -> None:
    """Initialize the weights and biases of a module using a uniform distribution.

    Args:
        module (nn.Module): The module to initialize.
        a (int | float): The lower bound of the uniform distribution. Default is 0.
        b (int | float): The upper bound of the uniform distribution. Default is 1.
        bias (int | float): The bias value. Default is 0.
    """
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.uniform_(module.weight, a, b)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(
    module: nn.Module,
    a: float = 0,
    mode: str = "fan_out",
    nonlinearity: str = "relu",
    bias: float = 0,
    distribution: str = "normal",
) -> None:
    """Initialize the weights and biases of a module using the Kaiming initialization method.

    Copied from mmengine.model.weight_init.kaiming_init

    Args:
        module (nn.Module): The module to initialize.
        a (float): The negative slope of the rectifier used after this layer (only used with 'leaky_relu' nonlinearity).
            Default is 0.
        mode (str): Either 'fan_in' (default) or 'fan_out'. Choosing 'fan_in' preserves the magnitude of the variance
            of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes in the backward pass.
            Default is 'fan_out'.
        nonlinearity (str): The non-linear function (nn.functional name), recommended to use 'relu' or 'leaky_relu'.
            Default is 'relu'.
        bias (float): The bias value. Default is 0.
        distribution (str): The type of distribution to use for weight initialization,
            either 'normal' (default) or 'uniform'.

    Returns:
        Tensor: The initialized tensor.
    """
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def bias_init_with_prob(prior_prob: float) -> float:
    """Initialize conv/fc bias value according to a given probability value."""
    return float(-np.log((1 - prior_prob) / prior_prob))


def _get_bases_name(m: nn.Module) -> list:
    return [b.__name__ for b in m.__class__.__bases__]


class BaseInit:
    """Base Init class."""

    def __init__(self, *, bias: int | float = 0, bias_prob: float | None = None, layer: str | list | None = None):
        self.wholemodule = False
        if not isinstance(bias, (int, float)):
            msg = f"bias must be a number, but got a {type(bias)}"
            raise TypeError(msg)

        if bias_prob is not None and not isinstance(bias_prob, float):
            msg = f"bias_prob type must be float, \
                    but got {type(bias_prob)}"
            raise TypeError(msg)

        if layer is not None:
            if not isinstance(layer, (str, list)):
                msg = f"layer must be a str or a list of str, \
                    but got a {type(layer)}"
                raise TypeError(msg)
        else:
            layer = []

        if bias_prob is not None:
            self.bias = bias_init_with_prob(bias_prob)
        else:
            self.bias = bias
        self.layer = [layer] if isinstance(layer, str) else layer

    def _get_init_info(self) -> str:
        return f"{self.__class__.__name__}, bias={self.bias}"


class ConstantInit(BaseInit):
    """Initialize module parameters with constant values.

    Args:
        val (int | float): the value to fill the weights in the module with
        bias (int | float): the value to fill the bias. Defaults to 0.
        bias_prob (float, optional): the probability for bias initialization.
            Defaults to None.
        layer (str | list[str], optional): the layer will be initialized.
            Defaults to None.
    """

    def __init__(self, val: int | float, **kwargs):
        super().__init__(**kwargs)
        self.val = val

    def __call__(self, module: nn.Module) -> None:
        """Initialize the module parameters.

        Args:
            module (nn.Module): The module to initialize.

        Returns:
            None
        """

        def init(m: nn.Module) -> None:
            if self.wholemodule:
                constant_init(m, self.val, self.bias)
            else:
                layername = m.__class__.__name__
                basesname = _get_bases_name(m)
                if len(set(self.layer) & {layername, *basesname}):
                    constant_init(m, self.val, self.bias)

        module.apply(init)
        if hasattr(module, "_params_init_info"):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self) -> str:
        return f"{self.__class__.__name__}: val={self.val}, bias={self.bias}"


class XavierInit(BaseInit):
    r"""Initialize module parameters with values according to the method described in the paper below.

    `Understanding the difficulty of training deep feedforward
    neural networks - Glorot, X. & Bengio, Y. (2010).
    <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_

    Args:
        gain (int | float): an optional scaling factor. Defaults to 1.
        bias (int | float): the value to fill the bias. Defaults to 0.
        bias_prob (float, optional): the probability for bias initialization.
            Defaults to None.
        distribution (str): distribution either be ``'normal'``
            or ``'uniform'``. Defaults to ``'normal'``.
        layer (str | list[str], optional): the layer will be initialized.
            Defaults to None.
    """

    def __init__(self, gain: int | float = 1, distribution: str = "normal", **kwargs):
        super().__init__(**kwargs)
        self.gain = gain
        self.distribution = distribution

    def __call__(self, module: nn.Module) -> None:
        """Initialize the module parameters.

        Args:
            module (nn.Module): The module to initialize.

        Returns:
            None
        """

        def init(m: nn.Module) -> None:
            if self.wholemodule:
                xavier_init(m, self.gain, self.bias, self.distribution)
            else:
                layername = m.__class__.__name__
                basesname = _get_bases_name(m)
                if len(set(self.layer) & {layername, *basesname}):
                    xavier_init(m, self.gain, self.bias, self.distribution)

        module.apply(init)
        if hasattr(module, "_params_init_info"):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self) -> str:
        return f"{self.__class__.__name__}: gain={self.gain}, distribution={self.distribution}, bias={self.bias}"


class NormalInit(BaseInit):
    r"""Initialize module parameters with the values drawn from the normal distribution.

    :math:`\mathcal{N}(\text{mean}, \text{std}^2)`.

    Args:
        mean (int | float):the mean of the normal distribution. Defaults to 0.
        std (int | float): the standard deviation of the normal distribution.
            Defaults to 1.
        bias (int | float): the value to fill the bias. Defaults to 0.
        bias_prob (float, optional): the probability for bias initialization.
            Defaults to None.
        layer (str | list[str], optional): the layer will be initialized.
            Defaults to None.
    """

    def __init__(self, mean: int | float = 0, std: int | float = 1, **kwargs):
        super().__init__(**kwargs)
        self.mean = mean
        self.std = std

    def __call__(self, module: nn.Module) -> None:
        """Initialize the module parameters.

        Args:
            module (nn.Module): The module to initialize.

        Returns:
            None
        """

        def init(m: nn.Module) -> None:
            if self.wholemodule:
                normal_init(m, self.mean, self.std, self.bias)
            else:
                layername = m.__class__.__name__
                basesname = _get_bases_name(m)
                if len(set(self.layer) & {layername, *basesname}):
                    normal_init(m, self.mean, self.std, self.bias)

        module.apply(init)
        if hasattr(module, "_params_init_info"):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self) -> str:
        return f"{self.__class__.__name__}: mean={self.mean}, std={self.std}, bias={self.bias}"


class TruncNormalInit(BaseInit):
    r"""Initialize module parameters with the values drawn from the normal distribution.

    :math:`\mathcal{N}(\text{mean}, \text{std}^2)` with values
    outside :math:`[a, b]`.

    Args:
        mean (float): the mean of the normal distribution. Defaults to 0.
        std (float):  the standard deviation of the normal distribution.
            Defaults to 1.
        a (float): The minimum cutoff value.
        b ( float): The maximum cutoff value.
        bias (float): the value to fill the bias. Defaults to 0.
        bias_prob (float, optional): the probability for bias initialization.
            Defaults to None.
        layer (str | list[str], optional): the layer will be initialized.
            Defaults to None.
    """

    def __init__(self, mean: float = 0, std: float = 1, a: float = -2, b: float = 2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mean = mean
        self.std = std
        self.a = a
        self.b = b

    def __call__(self, module: nn.Module) -> None:
        """Apply weight initialization to the given module.

        Args:
            module (nn.Module): The module to initialize.
        """

        def init(m: nn.Module) -> None:
            if self.wholemodule:
                trunc_normal_init(m, self.mean, self.std, self.a, self.b, self.bias)
            else:
                layername = m.__class__.__name__
                basesname = _get_bases_name(m)
                if len(set(self.layer) & {layername, *basesname}):
                    trunc_normal_init(m, self.mean, self.std, self.a, self.b, self.bias)

        module.apply(init)
        if hasattr(module, "_params_init_info"):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self) -> str:
        return f"{self.__class__.__name__}: a={self.a}, b={self.b}, mean={self.mean}, std={self.std}, bias={self.bias}"


class UniformInit(BaseInit):
    r"""Initialize module parameters with values drawn from the uniform distribution :math:`\mathcal{U}(a, b)`.

    Args:
        a (int | float): the lower bound of the uniform distribution.
            Defaults to 0.
        b (int | float): the upper bound of the uniform distribution.
            Defaults to 1.
        bias (int | float): the value to fill the bias. Defaults to 0.
        bias_prob (float, optional): the probability for bias initialization.
            Defaults to None.
        layer (str | list[str], optional): the layer will be initialized.
            Defaults to None.
    """

    def __init__(self, a: int | float = 0, b: int | float = 1, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b

    def __call__(self, module: nn.Module) -> None:
        """Apply weight initialization to the given module.

        Args:
            module (nn.Module): The module to initialize.
        """

        def init(m: nn.Module) -> None:
            if self.wholemodule:
                uniform_init(m, self.a, self.b, self.bias)
            else:
                layername = m.__class__.__name__
                basesname = _get_bases_name(m)
                if len(set(self.layer) & {layername, *basesname}):
                    uniform_init(m, self.a, self.b, self.bias)

        module.apply(init)
        if hasattr(module, "_params_init_info"):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self) -> str:
        return f"{self.__class__.__name__}: a={self.a}, b={self.b}, bias={self.bias}"


class KaimingInit(BaseInit):
    r"""Initialize module parameters with the values according to the method described in the paper below.

    `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification - He, K. et al. (2015).
    <https://www.cv-foundation.org/openaccess/content_iccv_2015/
    papers/He_Delving_Deep_into_ICCV_2015_paper.pdf>`_

    Args:
        a (int | float): the negative slope of the rectifier used after this
            layer (only used with ``'leaky_relu'``). Defaults to 0.
        mode (str):  either ``'fan_in'`` or ``'fan_out'``. Choosing
            ``'fan_in'`` preserves the magnitude of the variance of the weights
            in the forward pass. Choosing ``'fan_out'`` preserves the
            magnitudes in the backwards pass. Defaults to ``'fan_out'``.
        nonlinearity (str): the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` .
            Defaults to 'relu'.
        bias (int | float): the value to fill the bias. Defaults to 0.
        bias_prob (float, optional): the probability for bias initialization.
            Defaults to None.
        distribution (str): distribution either be ``'normal'`` or
            ``'uniform'``. Defaults to ``'normal'``.
        layer (str | list[str], optional): the layer will be initialized.
            Defaults to None.
    """

    def __init__(
        self,
        a: int | float = 0,
        mode: str = "fan_out",
        nonlinearity: str = "relu",
        distribution: str = "normal",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.a = a
        self.mode = mode
        self.nonlinearity = nonlinearity
        self.distribution = distribution

    def __call__(self, module: nn.Module) -> None:
        """Initialize the module parameters.

        Args:
            module (nn.Module): The module to initialize.
        """

        def init(m: nn.Module) -> None:
            if self.wholemodule:
                kaiming_init(m, self.a, self.mode, self.nonlinearity, self.bias, self.distribution)
            else:
                layername = m.__class__.__name__
                basesname = _get_bases_name(m)
                if len(set(self.layer) & {layername, *basesname}):
                    kaiming_init(m, self.a, self.mode, self.nonlinearity, self.bias, self.distribution)

        module.apply(init)
        if hasattr(module, "_params_init_info"):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self) -> str:
        return (
            f"{self.__class__.__name__}: a={self.a}, mode={self.mode}, "
            f"nonlinearity={self.nonlinearity}, "
            f"distribution ={self.distribution}, bias={self.bias}"
        )


class PretrainedInit:
    """Initialize module by loading a pretrained model.

    Args:
        checkpoint (str): the checkpoint file of the pretrained model should
            be load.
        prefix (str, optional): the prefix of a sub-module in the pretrained
            model. it is for loading a part of the pretrained model to
            initialize. For example, if we would like to only load the
            backbone of a detector model, we can set ``prefix='backbone.'``.
            Defaults to None.
        map_location (str): map tensors into proper locations. Defaults to cpu.
    """

    def __init__(self, checkpoint: str, prefix: str | None = None, map_location: str = "cpu"):
        self.checkpoint = checkpoint
        self.prefix = prefix
        self.map_location = map_location

    def __call__(self, module: nn.Module) -> None:
        """Initialize the module parameters by loading a pretrained model.

        Args:
            module (nn.Module): The module to initialize.
        """
        from otx.algo.utils.mmengine_utils import load_checkpoint_to_model, load_from_http, load_state_dict

        if self.prefix is None:
            if Path(self.checkpoint).exists():
                checkpoint = torch.load(self.checkpoint, map_location=self.map_location)
            elif self.checkpoint.startswith("http"):
                checkpoint = load_from_http(self.checkpoint)
            if checkpoint is not None:
                load_checkpoint_to_model(module, checkpoint)
                logger.info(f"load model from: {self.checkpoint}")
        else:
            logger.info(f"load {self.prefix} in model from: {self.checkpoint}")
            checkpoint = torch.load(self.checkpoint, map_location=self.map_location)
            state_dict = checkpoint.get("state_dict", checkpoint)
            prefix = self.prefix
            if not prefix.endswith("."):
                prefix += "."
            prefix_len = len(prefix)

            state_dict = {k[prefix_len:]: v for k, v in state_dict.items() if k.startswith(prefix)}

            load_state_dict(module, state_dict, strict=False)

        if hasattr(module, "_params_init_info"):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self) -> str:
        return f"{self.__class__.__name__}: load from {self.checkpoint}"


WEIGHT_INITIALIZERS = {
    "Constant": ConstantInit,
    "Xavier": XavierInit,
    "Normal": NormalInit,
    "TruncNormal": TruncNormalInit,
    "Uniform": UniformInit,
    "Kaiming": KaimingInit,
    "Pretrained": PretrainedInit,
}


def _initialize(module: nn.Module, cfg: dict, wholemodule: bool = False) -> None:
    cfg_copy = copy.deepcopy(cfg)
    initialize_type = cfg_copy.pop("type")
    func = WEIGHT_INITIALIZERS[initialize_type](**cfg_copy)
    # wholemodule flag is for override mode, there is no layer key in override
    # and initializer will give init values for the whole module with the name
    # in override.
    func.wholemodule = wholemodule
    func(module)


def _initialize_override(module: nn.Module, override: dict | list[dict], cfg: dict) -> None:
    if not isinstance(override, (dict, list)):
        msg = f"override must be a dict or a list of dict, \
                but got {type(override)}"
        raise TypeError(msg)

    override = [override] if isinstance(override, dict) else override

    for override_ in override:
        cp_override = copy.deepcopy(override_)
        name = cp_override.pop("name", None)
        if name is None:
            msg = f'`override` must contain the key "name", but got {cp_override}'
            raise ValueError(msg)
        # if override only has name key, it means use args in init_cfg
        if not cp_override:
            cp_override.update(cfg)
        # if override has name key and other args except type key, it will
        # raise error
        elif "type" not in cp_override:
            msg = f'`override` need "type" key, but got {cp_override}'
            raise ValueError(msg)

        if hasattr(module, name):
            _initialize(getattr(module, name), cp_override, wholemodule=True)
        else:
            msg = f"module did not have attribute {name}, but init_cfg is {cp_override}."
            raise RuntimeError(msg)


def initialize(module: nn.Module, init_cfg: dict | list[dict]) -> None:
    r"""Initialize a module.

    Args:
        module (``torch.nn.Module``): the module will be initialized.
        init_cfg (dict | list[dict]): initialization configuration dict to
            define initializer. OpenMMLab has implemented 6 initializers
            including ``Constant``, ``Xavier``, ``Normal``, ``Uniform``,
            ``Kaiming``, and ``Pretrained``.

    Example:
        >>> module = nn.Linear(2, 3, bias=True)
        >>> init_cfg = dict(type='Constant', layer='Linear', val =1 , bias =2)
        >>> initialize(module, init_cfg)
        >>> module = nn.Sequential(nn.Conv1d(3, 1, 3), nn.Linear(1,2))
        >>> # define key ``'layer'`` for initializing layer with different
        >>> # configuration
        >>> init_cfg = [dict(type='Constant', layer='Conv1d', val=1),
                dict(type='Constant', layer='Linear', val=2)]
        >>> initialize(module, init_cfg)
        >>> # define key``'override'`` to initialize some specific part in
        >>> # module
        >>> class FooNet(nn.Module):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.feat = nn.Conv2d(3, 16, 3)
        >>>         self.reg = nn.Conv2d(16, 10, 3)
        >>>         self.cls = nn.Conv2d(16, 5, 3)
        >>> model = FooNet()
        >>> init_cfg = dict(type='Constant', val=1, bias=2, layer='Conv2d',
        >>>     override=dict(type='Constant', name='reg', val=3, bias=4))
        >>> initialize(model, init_cfg)
        >>> model = ResNet(depth=50)
        >>> # Initialize weights with the pretrained model.
        >>> init_cfg = dict(type='Pretrained',
                checkpoint='torchvision://resnet50')
        >>> initialize(model, init_cfg)
        >>> # Initialize weights of a sub-module with the specific part of
        >>> # a pretrained model by using "prefix".
        >>> url = 'http://download.openmmlab.com/mmdetection/v2.0/retinanet/'\
        >>>     'retinanet_r50_fpn_1x_coco/'\
        >>>     'retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
        >>> init_cfg = dict(type='Pretrained',
                checkpoint=url, prefix='backbone.')
    """
    if not isinstance(init_cfg, (dict, list)):
        msg = f"init_cfg must be a dict or a list of dict, \
                but got {type(init_cfg)}"
        raise TypeError(msg)

    if isinstance(init_cfg, dict):
        init_cfg = [init_cfg]

    for cfg in init_cfg:
        # should deeply copy the original config because cfg may be used by
        # other modules, e.g., one init_cfg shared by multiple bottleneck
        # blocks, the expected cfg will be changed after pop and will change
        # the initialization behavior of other modules
        cp_cfg = copy.deepcopy(cfg)
        override = cp_cfg.pop("override", None)
        _initialize(module, cp_cfg)

        if override is not None:
            cp_cfg.pop("layer", None)
            _initialize_override(module, override, cp_cfg)
        else:
            # All attributes in module have same initialization.
            pass


def _no_grad_trunc_normal_(tensor: Tensor, mean: float, std: float, a: float, b: float) -> Tensor:
    # Method based on
    # https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    # Modified from
    # https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py
    def norm_cdf(x: float) -> float:
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [lower, upper], then translate
        # to [2lower-1, 2upper-1].
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0) -> Tensor:
    r"""Fills the input Tensor with values drawn from a truncated normal distribution.

    The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Modified from
    https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py

    Args:
        tensor (``torch.Tensor``): an n-dimensional `torch.Tensor`.
        mean (float): the mean of the normal distribution.
        std (float): the standard deviation of the normal distribution.
        a (float): the minimum cutoff value.
        b (float): the maximum cutoff value.
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
