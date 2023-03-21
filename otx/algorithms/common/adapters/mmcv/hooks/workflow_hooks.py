# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import copy
import datetime
import json

from mmcv.utils import Registry

from otx.mpa.utils.logger import get_logger

logger = get_logger()
WORKFLOW_HOOKS = Registry("workflow_hooks")


def build_workflow_hook(config, *args, **kwargs):
    logger.info(f"called build_workflow_hook({config})")
    whook_type = config.pop("type")
    # event = config.pop('event')
    if whook_type not in WORKFLOW_HOOKS:
        raise KeyError(f"not supported workflow hook type {whook_type}")
    else:
        whook_cls = WORKFLOW_HOOKS.get(whook_type)
    return whook_cls(*args, **kwargs, **config)


class WorkflowHook:
    def __init__(self, name):
        self.name = name

    def before_workflow(self, workflow, idx=-1, results=None):
        pass

    def after_workflow(self, workflow, idx=-1, results=None):
        pass

    def before_stage(self, workflow, idx, results=None):
        pass

    def after_stage(self, workflow, idx, results=None):
        pass


@WORKFLOW_HOOKS.register_module()
class SampleLoggingHook(WorkflowHook):
    def __init__(self, name=__name__, log_level="DEBUG"):
        super(SampleLoggingHook, self).__init__(name)
        self.logging = getattr(logger, log_level.lower())

    def before_stage(self, wf, stage_idx, results):
        self.logging(f"called {self.name}.run()")
        self.logging(f"stage index {stage_idx}, results keys = {results.keys()}")
        result_key = f"{self.name}|{stage_idx}"
        results[result_key] = dict(message=f"this is a sample result of the {__name__} hook")


@WORKFLOW_HOOKS.register_module()
class WFProfileHook(WorkflowHook):
    def __init__(self, name=__name__, output_path=None):
        super(WFProfileHook, self).__init__(name)
        self.output_path = output_path
        self.profile = dict(start=0, end=0, elapsed=0, stages=dict())
        logger.info(f"initialized {__name__}....")

    def before_workflow(self, wf, idx=-1, results=None):
        self.profile["start"] = datetime.datetime.now()

    def after_workflow(self, wf, idx=-1, results=None):
        self.profile["end"] = datetime.datetime.now()
        self.profile["elapsed"] = self.profile["end"] - self.profile["start"]

        str_dumps = json.dumps(self.profile, indent=2, default=str)
        logger.info("** workflow profile results **")
        logger.info(str_dumps)
        if self.output_path is not None:
            with open(self.output_path, "w") as f:
                f.write(str_dumps)

    def before_stage(self, wf, idx=-1, results=None):
        stages = self.profile.get("stages")
        stages[f"{idx}"] = {}
        stages[f"{idx}"]["start"] = datetime.datetime.now()

    def after_stage(self, wf, idx=-1, results=None):
        stages = self.profile.get("stages")
        stages[f"{idx}"]["end"] = datetime.datetime.now()
        stages[f"{idx}"]["elapsed"] = stages[f"{idx}"]["end"] - stages[f"{idx}"]["start"]


@WORKFLOW_HOOKS.register_module()
class AfterStageWFHook(WorkflowHook):
    def __init__(self, name, stage_cfg_updated_callback):
        self.callback = stage_cfg_updated_callback
        super().__init__(name)

    def after_stage(self, workflow, idx, results=None):
        logger.info(f"{__name__}: called after_stage()")
        name = copy.deepcopy(workflow.stages[idx].name)
        cfg = copy.deepcopy(workflow.stages[idx].cfg)
        self.callback(name, cfg)
