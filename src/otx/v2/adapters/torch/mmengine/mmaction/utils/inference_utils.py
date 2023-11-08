"""OTX adapters.torch.mmengine utils."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
import numpy as np
import torch
from datetime import datetime
from typing import Sequence, Union, Iterable, Dict, List, Tuple, Optional, Callable 
import mmengine
from mmengine.config import Config, ConfigDict
from mmengine.dataset import pseudo_collate
from mmengine.dataset import Compose
from mmengine.fileio import (get_file_backend, isdir, join_path,
                             list_dir_or_file)
from mmengine.registry import FUNCTIONS, VISUALIZERS, DefaultScope
from mmengine.visualization import Visualizer
from mmengine.structures import InstanceData

from mmaction.utils import ConfigType, get_str_type
from mmaction.structures import ActionDataSample

InstanceList = List[InstanceData]
InputType = Union[str, np.ndarray, torch.Tensor]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[InstanceData, InstanceList]
ResType = Union[Dict, List[Dict], InstanceData, List[InstanceData]]
ConfigType = Union[Config, ConfigDict]

def inputs_to_list(inputs: InputsType) -> list:
    """Preprocess the inputs to a list.

    Preprocess inputs to a list according to its type:

    - list or tuple: return inputs
    - str:
        - Directory path: return all files in the directory
        - other cases: return a list containing the string. The string
            could be a path to file, a url or other types of string according
            to the task.

    Args:
        inputs (InputsType): Inputs for the inferencer.

    Returns:
        list: List of input for the :meth:`preprocess`.
    """
    if isinstance(inputs, str):
        backend = get_file_backend(inputs)
        if hasattr(backend, 'isdir') and isdir(inputs):
            # Backends like HttpsBackend do not implement `isdir`, so only
            # those backends that implement `isdir` could accept the inputs
            # as a directory
            filename_list = list_dir_or_file(inputs, list_dir=False)
            inputs = [
                join_path(inputs, filename) for filename in filename_list
            ]

    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    return list(inputs)

def get_chunk_data(inputs: Iterable, chunk_size: int):
    """Get batch data from dataset.

    Args:
        inputs (Iterable): An iterable dataset.
        chunk_size (int): Equivalent to batch size.

    Yields:
        list: batch data.
    """
    inputs_iter = iter(inputs)
    breakpoint()
    while True:
        try:
            chunk_data = []
            for _ in range(chunk_size):
                processed_data = next(inputs_iter)
                chunk_data.append(processed_data)
            yield chunk_data
        except StopIteration:
            if chunk_data:
                yield chunk_data
            break
    
def preprocess(inputs: InputsType, pipeline:Union[Dict, List], collate_fn:Callable, batch_size: int = 1, **kwargs):
    """Process the inputs into a model-feedable format.

    Customize your preprocess by overriding this method. Preprocess should
    return an iterable object, of which each item will be used as the
    input of ``model.test_step``.

    ``BaseInferencer.preprocess`` will return an iterable chunked data,
    which will be used in __call__ like this:

    .. code-block:: python

        def __call__(self, inputs, batch_size=1, **kwargs):
            chunked_data = self.preprocess(inputs, batch_size, **kwargs)
            for batch in chunked_data:
                preds = self.forward(batch, **kwargs)

    Args:
        inputs (InputsType): Inputs given by user.
        batch_size (int): batch size. Defaults to 1.

    Yields:
        Any: Data processed by the ``pipeline`` and ``collate_fn``.
    """
    chunked_data = get_chunk_data(
        map(pipeline, inputs), batch_size)
    yield from map(collate_fn, chunked_data)
    
def visualize(
    visualizer,
    inputs: InputsType,
    preds: PredType,
    return_vis: bool = False,
    show: bool = False,
    wait_time: int = 0,
    draw_pred: bool = True,
    fps: int = 30,
    out_type: str = 'video',
    target_resolution: Optional[Tuple[int]] = None,
    vid_out_dir: str = '',
) -> Union[List[np.ndarray], None]:
    """Visualize predictions.

    Args:
        inputs (List[Union[str, np.ndarray]]): Inputs for the inferencer.
        preds (List[Dict]): Predictions of the model.
        return_vis (bool): Whether to return the visualization result.
            Defaults to False.
        show (bool): Whether to display the image in a popup window.
            Defaults to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        draw_pred (bool): Whether to draw prediction labels.
            Defaults to True.
        fps (int): Frames per second for saving video. Defaults to 4.
        out_type (str): Output format type, choose from 'img', 'gif',
            'video'. Defaults to ``'img'``.
        target_resolution (Tuple[int], optional): Set to
            (desired_width desired_height) to have resized frames. If
            either dimension is None, the frames are resized by keeping
            the existing aspect ratio. Defaults to None.
        vid_out_dir (str): Output directory of visualization results.
            If left as empty, no file will be saved. Defaults to ''.

    Returns:
        List[np.ndarray] or None: Returns visualization results only if
        applicable.
    """
    if visualizer is None or (not show and vid_out_dir == ''
                                    and not return_vis):
        return None

    results = []
    num_visualized_vids = 0
    for single_input, pred in zip(inputs, preds):
        if isinstance(single_input, str):
            frames = single_input
            video_name = osp.basename(single_input)
        elif isinstance(single_input, np.ndarray):
            frames = single_input.copy()
            video_num = str(num_visualized_vids).zfill(8)
            video_name = f'{video_num}.mp4'
        else:
            raise ValueError('Unsupported input type: '
                                f'{type(single_input)}')

        out_path = osp.join(vid_out_dir, video_name) if vid_out_dir != '' \
            else None

        visualization = visualizer.add_datasample(
            video_name,
            frames,
            pred,
            show_frames=show,
            wait_time=wait_time,
            draw_gt=False,
            draw_pred=draw_pred,
            fps=fps,
            out_type=out_type,
            out_path=out_path,
            target_resolution=target_resolution,
        )
        results.append(visualization)
        num_visualized_vids += 1

    return results

def postprocess(
    preds: PredType,
    visualization: Optional[List[np.ndarray]] = None,
    return_datasample: bool = False,
    print_result: bool = False,
    pred_out_file: str = '',
) -> Union[ResType, Tuple[ResType, np.ndarray]]:
    """Process the predictions and visualization results from ``forward``
    and ``visualize``.

    This method should be responsible for the following tasks:

    1. Convert datasamples into a json-serializable dict if needed.
    2. Pack the predictions and visualization results and return them.
    3. Dump or log the predictions.

    Args:
        preds (List[Dict]): Predictions of the model.
        visualization (Optional[np.ndarray]): Visualized predictions.
        return_datasample (bool): Whether to use Datasample to store
            inference results. If False, dict will be used.
        print_result (bool): Whether to print the inference result w/o
            visualization to the console. Defaults to False.
        pred_out_file: File to save the inference results w/o
            visualization. If left as empty, no file will be saved.
            Defaults to ''.

    Returns:
        dict: Inference and visualization results with key ``predictions``
        and ``visualization``.

        - ``visualization`` (Any): Returned by :meth:`visualize`.
        - ``predictions`` (dict or DataSample): Returned by
            :meth:`forward` and processed in :meth:`postprocess`.
            If ``return_datasample=False``, it usually should be a
            json-serializable dict containing only basic data elements such
            as strings and numbers.
    """
    result_dict = {}
    results = preds
    if not return_datasample:
        results = []
        for pred in preds:
            result = pred2dict(pred)
            results.append(result)
    # Add video to the results after printing and dumping
    result_dict['predictions'] = results
    if print_result:
        print(result_dict)
    if pred_out_file != '':
        mmengine.dump(result_dict, pred_out_file)
    result_dict['visualization'] = visualization
    return result_dict

def pred2dict(data_sample: ActionDataSample) -> Dict:
    """Extract elements necessary to represent a prediction into a
    dictionary. It's better to contain only basic data elements such as
    strings and numbers in order to guarantee it's json-serializable.

    Args:
        data_sample (ActionDataSample): The data sample to be converted.

    Returns:
        dict: The output dictionary.
    """
    result = {}
    result['pred_labels'] = data_sample.pred_labels.item.tolist()
    result['pred_scores'] = data_sample.pred_scores.item.tolist()
    return result

def init_visualizer(cfg: ConfigType) -> Optional[Visualizer]:
    """Initialize visualizers.

    Args:
        cfg (ConfigType): Config containing the visualizer information.

    Returns:
        Visualizer or None: Visualizer initialized with config.
    """
    if 'visualizer' not in cfg:
        return None
    timestamp = str(datetime.timestamp(datetime.now()))
    name = cfg.visualizer.get('name', timestamp)
    if Visualizer.check_instance_created(name):
        name = f'{name}-{timestamp}'
    cfg.visualizer.name = name
    return VISUALIZERS.build(cfg.visualizer)
    
def init_collate(cfg: ConfigType) -> Callable:
    """Initialize the ``collate_fn`` with the given config.

    The returned ``collate_fn`` will be used to collate the batch data.
    If will be used in :meth:`preprocess` like this

    .. code-block:: python
        def preprocess(self, inputs, batch_size, **kwargs):
            ...
            dataloader = map(self.collate_fn, dataloader)
            yield from dataloader

    Args:
        cfg (ConfigType): Config which could contained the `collate_fn`
            information. If `collate_fn` is not defined in config, it will
            be :func:`pseudo_collate`.

    Returns:
        Callable: Collate function.
    """
    try:
        with FUNCTIONS.switch_scope_and_registry('mmaction') as registry:
            collate_fn = registry.get(cfg.test_dataloader.collate_fn)
    except AttributeError:
        collate_fn = pseudo_collate
    return collate_fn  # type: ignore
    
def init_pipeline(cfg: ConfigType) -> Compose:
    """Initialize the test pipeline."""
    test_pipeline = cfg.test_dataloader.dataset.pipeline
    # Alter data pipelines for decode
    if 'Init' in get_str_type(test_pipeline[0]['type']):
        test_pipeline = test_pipeline[1:]
    for i in range(len(test_pipeline)):
        if 'Decode' in get_str_type(test_pipeline[i]['type']):
            test_pipeline[i] = dict(type='RawFrameDecode')
    # Alter data pipelines to close TTA, avoid OOM
    # Use center crop instead of multiple crop
    for i in range(len(test_pipeline)):
        if get_str_type(
                test_pipeline[i]['type']) in ['ThreeCrop', 'TenCrop']:
            test_pipeline[i]['type'] = 'CenterCrop'
    # Use single clip for `Recognizer3D`
    if cfg.model.type == 'Recognizer3D':
        for i in range(len(test_pipeline)):
            if get_str_type(test_pipeline[i]['type']) == 'SampleFrames':
                test_pipeline[i]['num_clips'] = 1
    # Pack multiple types of input format
    test_pipeline.insert(
        0,
        dict(
            type='InferencerPackInput',
            input_format="rawframes"
        )
    )

    return Compose(test_pipeline)