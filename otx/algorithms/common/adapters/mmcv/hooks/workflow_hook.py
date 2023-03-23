"""Workflow hooks."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import copy
import datetime
import json

from mmcv.utils import Registry

from otx.algorithms.common.utils.logger import get_logger

logger = get_logger()
WORKFLOW_HOOKS = Registry("workflow_hooks")

# pylint: disable=unused-argument


def build_workflow_hook(config, *args, **kwargs):
    """Build a workflow hook."""
    logger.info(f"called build_workflow_hook({config})")
    whook_type = config.pop("type")
    # event = config.pop('event')
    if whook_type not in WORKFLOW_HOOKS:
        raise KeyError(f"not supported workflow hook type {whook_type}")
    whook_cls = WORKFLOW_HOOKS.get(whook_type)
    return whook_cls(*args, **kwargs, **config)


class WorkflowHook:
    """Workflow hook."""

    def __init__(self, name):
        self.name = name

    def before_workflow(self, workflow, idx=-1, results=None):
        """Before workflow."""
        return

    def after_workflow(self, workflow, idx=-1, results=None):
        """After workflow."""
        return

    def before_stage(self, workflow, idx, results=None):
        """Before stage."""
        return

    def after_stage(self, workflow, idx, results=None):
        """After stage."""
        return


@WORKFLOW_HOOKS.register_module()
class SampleLoggingHook(WorkflowHook):
    """Sample logging hook."""

    def __init__(self, name=__name__, log_level="DEBUG"):
        super().__init__(name)
        self.logging = getattr(logger, log_level.lower())

    def before_stage(self, workflow, idx, results=None):
        """Before stage."""
        self.logging(f"called {self.name}.run()")
        self.logging(f"stage index {idx}, results keys = {results.keys()}")
        result_key = f"{self.name}|{idx}"
        results[result_key] = dict(message=f"this is a sample result of the {__name__} hook")


@WORKFLOW_HOOKS.register_module()
class WFProfileHook(WorkflowHook):
    """Workflow profile hook."""

    def __init__(self, name=__name__, output_path=None):
        super().__init__(name)
        self.output_path = output_path
        self.profile = dict(start=0, end=0, elapsed=0, stages=dict())
        logger.info(f"initialized {__name__}....")

    def before_workflow(self, workflow, idx=-1, results=None):
        """Before workflow."""
        self.profile["start"] = datetime.datetime.now()

    def after_workflow(self, workflow, idx=-1, results=None):
        """After workflow."""
        self.profile["end"] = datetime.datetime.now()
        self.profile["elapsed"] = self.profile["end"] - self.profile["start"]

        str_dumps = json.dumps(self.profile, indent=2, default=str)
        logger.info("** workflow profile results **")
        logger.info(str_dumps)
        if self.output_path is not None:
            with open(self.output_path, "w") as f:  # pylint: disable=unspecified-encoding
                f.write(str_dumps)

    def before_stage(self, workflow, idx=-1, results=None):
        """Before stage."""
        stages = self.profile.get("stages")
        stages[f"{idx}"] = {}
        stages[f"{idx}"]["start"] = datetime.datetime.now()

    def after_stage(self, workflow, idx=-1, results=None):
        """After stage."""
        stages = self.profile.get("stages")
        stages[f"{idx}"]["end"] = datetime.datetime.now()
        stages[f"{idx}"]["elapsed"] = stages[f"{idx}"]["end"] - stages[f"{idx}"]["start"]


@WORKFLOW_HOOKS.register_module()
class AfterStageWFHook(WorkflowHook):
    """After stage workflow hook."""

    def __init__(self, name, stage_cfg_updated_callback):
        self.callback = stage_cfg_updated_callback
        super().__init__(name)

    def after_stage(self, workflow, idx, results=None):
        """After stage."""
        logger.info(f"{__name__}: called after_stage()")
        name = copy.deepcopy(workflow.stages[idx].name)
        cfg = copy.deepcopy(workflow.stages[idx].cfg)
        self.callback(name, cfg)
