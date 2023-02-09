import multiprocessing
import os
from abc import ABC, abstractmethod
from functools import partial
from multiprocessing import Process
from typing import Any, Callable, Dict, Optional, Union

from hpopt.hpo_base import HpoBase, Trial, TrialStatus
from hpopt.logger import get_logger
from hpopt.utils import check_positive

import torch

logger = get_logger()


class ResourceManager(ABC):
    @abstractmethod
    def reserve_resource(self, trial_id):
        raise NotImplementedError

    @abstractmethod
    def release_resource(self, trial_id):
        raise NotImplementedError

    @abstractmethod
    def have_available_resource(self):
        raise NotImplementedError

class CPUResourceManager(ResourceManager):
    def __init__(self, num_parallel_trial: int = 4):
        check_positive(num_parallel_trial, "num_parallel_trial")

        self._num_parallel_trial = num_parallel_trial
        self._usage_status = []

    def reserve_resource(self, trial_id: Any):
        if not self.have_available_resource():
            return None
        if trial_id in self._usage_status:
            raise RuntimeError(f"{trial_id} already has reserved resource.")

        logger.debug(f"{trial_id} reserved.")
        self._usage_status.append(trial_id)
        return {}

    def release_resource(self, trial_id: Any):
        if trial_id not in self._usage_status:
            logger.warning(f"{trial_id} trial don't use resource now.")
        else:
            self._usage_status.remove(trial_id)
            logger.debug(f"{trial_id} released.")

    def have_available_resource(self):
        return len(self._usage_status) < self._num_parallel_trial

class GPUResourceManager(ResourceManager):
    def __init__(self, num_gpu_for_single_trial: int = 1, available_gpu: Optional[str] = None):
        check_positive(num_gpu_for_single_trial, "num_gpu_for_single_trial")

        self._num_gpu_for_single_trial = num_gpu_for_single_trial
        self._available_gpu = self._set_available_gpu(available_gpu)
        self._usage_status = {}

    def _set_available_gpu(self, available_gpu: Optional[str] = None):
        if available_gpu is None:
            if os.getenv("CUDA_VISIBLE_DEVICES") is not None:
                available_gpu = self._transform_gpu_format_from_string_to_arr(os.getenv("CUDA_VISIBLE_DEVICES"))
            else:
                num_gpus = torch.cuda.device_count()
                available_gpu = [val for val in range(num_gpus)]
        else:
            available_gpu = self._transform_gpu_format_from_string_to_arr(available_gpu)

        return available_gpu

    def _transform_gpu_format_from_string_to_arr(self, gpu: str):
        for val in gpu.split(','):
            if not val.isnumeric():
                raise ValueError(
                    "gpu format is wrong. "
                    "gpu should only have numbers delimited by ','.\n"
                    f"your value is {gpu}"
                )
        return [int(val) for val in gpu.split(',')]

    def reserve_resource(self, trial_id):
        if not self.have_available_resource():
            return None
        if trial_id in self._usage_status:
            raise RuntimeError(f"{trial_id} already has reserved resource.")

        resource = list(self._available_gpu[:self._num_gpu_for_single_trial])
        self._available_gpu = self._available_gpu[self._num_gpu_for_single_trial:]

        self._usage_status[trial_id] = resource
        return {"CUDA_VISIBLE_DEVICES" : ",".join([str(val) for val in resource])}


    def release_resource(self, trial_id):
        if trial_id not in self._usage_status:
            logger.warning(f"{trial_id} trial don't use resource now.")
        else:
            self._available_gpu.extend(self._usage_status[trial_id])
            del self._usage_status[trial_id]

    def have_available_resource(self):
        return len(self._available_gpu) >= self._num_gpu_for_single_trial

def get_resource_manager(
    resource_type: str,
    num_parallel_trial: Optional[int] = None,
    num_gpu_for_single_trial: Optional[int] = None,
    available_gpu: Optional[str] = None,
):
    if resource_type == "gpu" and not torch.cuda.is_available():
        logger.warning("GPU can't be used now. resource type is modified to cpu.")
        resource_type = "cpu"

    if resource_type == "cpu":
        args = {"num_parallel_trial" : num_parallel_trial}
        args = _remove_none_from_dict(args)
        return CPUResourceManager(**args)
    elif resource_type == "gpu":
        args = {"num_gpu_for_single_trial" : num_gpu_for_single_trial, "available_gpu" : available_gpu}
        args = _remove_none_from_dict(args)
        return GPUResourceManager(**args)
    else:
        raise ValueError(f"Available resource type is cpu, gpu. Your value is {resource_type}.")

def _remove_none_from_dict(d: Dict):
    key_to_remove = [key for key, val in d.items() if val is None] 
    for key in key_to_remove:
        del d[key]
    return d

class HpoLoop:
    def __init__(
        self,
        hpo_algo: HpoBase,
        train_func: Callable,
        resource_type: str = "gpu",
        num_parallel_trial: Optional[int] = None,
        num_gpu_for_single_trial: Optional[int] = None,
        available_gpu: Optional[str] = None,
    ):
        self._hpo_algo = hpo_algo
        self._train_func = train_func
        self._running_trials: Dict[int, Dict[str, Union[Trial, Process]]] = {}
        self._mp = multiprocessing.get_context("spawn")
        self._uid_index = 0
        self._resource_manager = get_resource_manager(
            resource_type,
            num_parallel_trial,
            num_gpu_for_single_trial,
            available_gpu
        )

    def run(self):
        logger.info("HPO loop starts.")
        while not self._hpo_algo.is_done():
            if self._resource_manager.have_available_resource():
                trial = self._hpo_algo.get_next_sample()
                if trial is not None:
                    self._start_trial_process(trial)

            self._remove_finished_process()
            self._get_reports()

        logger.info("HPO loop is done.")
        self._get_reports()
        self._join_all_processes()

    def _start_trial_process(self, trial: Trial):
        logger.info(f"{trial.id} trial is now running.")
        logger.debug(f"{trial.id} hyper paramter => {trial.configuration}")
        
        trial.status = TrialStatus.RUNNING
        uid = self._get_uid()

        origin_env = os.environ
        env = self._resource_manager.reserve_resource(uid)
        if env is not None:
            for key, val in env.items():
                os.environ[key] = val

        pipe1, pipe2 = self._mp.Pipe(True)
        process = self._mp.Process(
            target=_run_train,
            args=(
                self._train_func,
                trial.get_train_configuration(),
                partial(_report_score, pipe=pipe2, trial_id=trial.id)
            )
        )
        os.environ = origin_env
        self._running_trials[uid] = {"process" : process, "trial" : trial, "pipe" : pipe1}
        process.start()

    def _remove_finished_process(self):
        trial_to_remove = []
        for uid, val in self._running_trials.items():
            process = val["process"]
            if not process.is_alive():
                val["pipe"].close()
                process.join()
                trial_to_remove.append(uid)

        for uid in trial_to_remove:
            trial = self._running_trials[uid]["trial"]
            trial.status = TrialStatus.STOP
            self._resource_manager.release_resource(uid)
            del self._running_trials[uid]

    def _get_reports(self):
        for trial in self._running_trials.values():
            pipe = trial["pipe"]
            if pipe.poll():
                try:
                    report = pipe.recv()
                except EOFError:
                    continue
                trial_status = self._hpo_algo.report_score(
                    report["score"],
                    report["progress"],
                    report["trial_id"],
                    report["done"]
                )
                pipe.send(trial_status)

        self._hpo_algo.save_results()
    
    def _join_all_processes(self):
        for val in self._running_trials.values():
            val["pipe"].close()

        for val in self._running_trials.values():
            process = val["process"]
            process.join()

        self._running_trials = {}

    def _get_uid(self):
        uid = self._uid_index
        self._uid_index += 1
        return uid

def _run_train(train_func: Callable, hp_config: Dict, report_func: Callable):
    multiprocessing.set_start_method(None, True)  # set multi process method as default
    train_func(hp_config, report_func)

def _report_score(
    score: Union[int, float],
    progress: Union[int, float],
    pipe,
    trial_id: Any,
    done: bool = False
):
    logger.debug(f"score : {score}, progress : {progress}, trial_id : {trial_id}, pid : {os.getpid()}, done : {done}")
    try:
        pipe.send(
            {
                "score" : score,
                "progress" : progress,
                "trial_id" : trial_id,
                "pid" : os.getpid(),
                "done" : done
            }
        )
    except BrokenPipeError:
        return TrialStatus.STOP
    try:
        trial_status = pipe.recv()
    except EOFError:
        return TrialStatus.STOP

    logger.debug(f"trial_status : {trial_status}")
    return trial_status

def run_hpo_loop(
    hpo_algo: HpoBase,
    train_func: Callable,
    resource_type: str = "gpu",
    num_parallel_trial: Optional[int] = None,
    num_gpu_for_single_trial: Optional[int] = None,
    available_gpu: Optional[str] = None,
):
    hpo_loop = HpoLoop(hpo_algo, train_func, resource_type, num_parallel_trial, num_gpu_for_single_trial,available_gpu)
    hpo_loop.run()
