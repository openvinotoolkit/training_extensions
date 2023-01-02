# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os

import pandas as pd
import torch
from mmcv.runner import HOOKS, Hook

from otx.mpa.utils.logger import get_logger

from .utils import plot_mem, print_report

logger = get_logger()


@HOOKS.register_module()
class MemInspectHook(Hook):
    def __init__(self, **kwargs):
        super(MemInspectHook, self).__init__()
        self.exp = kwargs.get("exp", "baseline")
        self.print_report = kwargs.get("print_report", False)
        self.output_file = kwargs.get("output_file", f"gpu_mem_plot_{self.exp}.png")
        data_cfg = kwargs.get("data_cfg", None)
        if data_cfg is None:
            raise ValueError("cannot find data config")
        logger.info(f"data_cfg = {data_cfg}")

        self.data_args = self._parse_data_cfg(data_cfg)
        logger.info(f"keys in data args = {self.data_args.keys()}")

        # input value will be passed as positional argument to the model
        self.input = self.data_args.pop("input", None)
        if self.input is None:
            raise ValueError("invalid data configuration. 'input' key is the mandatory for the data configuration.")

    def _parse_data_cfg(self, cfg):
        input_args = {}
        if not isinstance(cfg, dict):
            raise ValueError("invalid configuration for the data. supported only dictionary type")
        for idx, (k, v) in enumerate(cfg.items()):
            type_ = v.get("type", None)
            args = v.get("args", None)
            if type_ is not None and args is not None:
                input_args[k] = getattr(torch, type_)(*args)
            elif isinstance(v, dict):
                input_args[k] = self._parse_data_cfg(v)

        return input_args

    # referenced from https://github.com/quentinf00/article-memory-log
    def _generate_mem_hook(self, mem, idx, hook_type, exp):
        def hook(self, *args):
            if len(mem) == 0 or mem[-1]["exp"] != exp:
                call_idx = 0
            else:
                call_idx = mem[-1]["call_idx"] + 1

            mem_all, mem_cached = torch.cuda.memory_allocated(), torch.cuda.memory_cached()
            torch.cuda.synchronize()
            mem.append(
                {
                    "layer_idx": idx,
                    "call_idx": call_idx,
                    "layer_type": type(self).__name__,
                    "exp": exp,
                    "hook_type": hook_type,
                    "mem_all": mem_all,
                    "mem_cached": mem_cached,
                }
            )

        return hook

    def before_run(self, runner):
        model = runner.model

        exp = "baseline"
        mem_log = []
        hooks = []
        for idx, module in enumerate(runner.model.modules()):
            hooks.append(module.register_forward_pre_hook(self._generate_mem_hook(mem_log, idx, "pre", exp)))
            hooks.append(module.register_forward_hook(self._generate_mem_hook(mem_log, idx, "fwd", exp)))
            hooks.append(module.register_backward_hook(self._generate_mem_hook(mem_log, idx, "bwd", exp)))

        try:
            out = model(self.input, **self.data_args)
            loss = out["loss"].sum()
            loss.backward()
        finally:
            [hook.remove() for hook in hooks]

        df = pd.DataFrame(mem_log)
        plot_mem(
            df, exps=[self.exp], output_file=os.path.join(runner.work_dir, self.output_file), normalize_call_idx=False
        )
        if self.print_report:
            print_report(df, exp=self.exp, logger=logger)
