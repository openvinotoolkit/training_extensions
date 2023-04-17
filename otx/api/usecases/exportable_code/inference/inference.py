"""Interface for inferencer."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import abc
import logging
import multiprocessing
import queue
import warnings
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple, Union

import numpy as np

# pylint: disable=no-name-in-module
from openvino.inference_engine import ExecutableNetwork, IECore, InferRequest
from openvino.inference_engine.constants import OK, RESULT_NOT_READY

from otx.api.entities.annotation import AnnotationSceneEntity
from otx.api.usecases.exportable_code.streamer.streamer import BaseStreamer

__all__ = [
    "AsyncOpenVINOTask",
    "BaseInferencer",
    "BaseOpenVINOInferencer",
    "IInferencer",
]

logger = logging.getLogger(__name__)


class IInferencer(metaclass=abc.ABCMeta):
    """Base interface class for the inference task.

    This class could be used by both the analyse method in the task, and the exportable code inference.

    """

    @abc.abstractmethod
    def pre_process(self, image: np.ndarray) -> Tuple[Any, Any]:
        """Pre-process input image and return the pre-processed image with meta-data if required.

        This method should pre-process the input image, and return the processed output and if required a Tuple with
        metadata that is required for post_process to work.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, image: Any) -> Any:
        """Forward the input image to the model and return the output.

        NOTE: The input is typed as Any at the moment, mainly because it could be numpy
            array,torch Tensor or tf Tensor. In the future, it could be an idea to be
            more specific.

        This method should perform the prediction by forward-passing the input image
        to the model, and return the predictions in a dictionary format.

        For instance, for a segmentation task, the predictions could be {"mask": mask}.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def post_process(self, prediction: Any, metadata: Any) -> Union[AnnotationSceneEntity, Tuple[Any, ...]]:
        """Post-process the raw predictions, and return the AnnotationSceneEntity.

        This method should include the post-processing methods that are applied to the raw predictions from the
        self.forward() stage.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, image: np.ndarray) -> Union[AnnotationSceneEntity, Tuple[Any, ...]]:
        """This method performs a prediction."""
        raise NotImplementedError


class BaseInferencer(IInferencer, abc.ABC):
    """Base class for standard inference.

    The user needs to implement the following:
        + `load_model`
        + `pre_process`
        + `forward`
        + `post_process`
    """

    def predict(self, image: np.ndarray) -> Union[AnnotationSceneEntity, Tuple[Any, ...]]:
        """Perform a prediction for a given input image.

        Args:
            image: Input image

        Returns:
            Output predictions
        """
        image, metadata = self.pre_process(image)
        predictions = self.forward(image)
        predictions = self.post_process(predictions, metadata)

        return predictions


class BaseOpenVINOInferencer(BaseInferencer, abc.ABC):
    """Base class for OpenVINO inference.

    Can handle the basic flow of reading and loading a model. If the network needs to be reshaped,
    override the load_model function.
    One would need to implement the following methods to use this as OpenVINO
    Inferencer
        + `pre_process`
        + `forward`
        + `post_process`

    Args:
        weight_path: Path to the weight file
        device: Device to use for inference. Check available devices
            with IECore().available_devices.
        num_requests: Number of simultaneous requests that can be issued
            to the model, has no effect on synchronous execution.

    Raises:
        ValueError: Raised if the device is not available.
    """

    def __init__(
        self,
        model_file: Union[str, bytes],
        weights_file: Union[str, bytes, None] = None,
        device: str = "CPU",
        num_requests: int = 1,
    ):
        self.ie_core = IECore()
        if device not in self.ie_core.available_devices:
            raise ValueError(
                f"Device '{device}' is not available for inference. Available devices "
                f"are: {self.ie_core.available_devices}"
            )

        self.device: str = device
        self.num_requests: int = num_requests

        self.input_keys: Optional[List[str]] = None
        self.output_keys: Optional[List[str]] = None

        self.net: Optional[ExecutableNetwork] = None
        self.model: ExecutableNetwork
        self.load_model(model_file, weights_file)

    def read_model(
        self,
        model_file: Union[Path, str, bytes],
        weights_file: Union[Path, str, bytes, None] = None,
    ):
        """Reads an OpenVINO model and saves its input and output keys to a list.

        Args:
            model_file: Path to the model file or bytes with data from OpenVINO's .xml file.
            weights: A .xml, .bin or .onnx file to be loaded. if a .xml
                or .bin is provided a file of the same name with the
                other extension is also expected

        Raises:
            ValueError: Raised if a weights file that is not compatible with OpenVINO
                        is provided
        """
        if isinstance(model_file, str):
            path = Path(model_file)
            if path.suffix == ".onnx":
                weights_file = None
            elif path.suffix in (".xml", ".bin"):
                model_file = path.with_suffix(".xml")
                weights_file = path.with_suffix(".bin")
            else:
                raise ValueError(f"Unsupported file extension: {path.suffix}")

        init_from_buffer = isinstance(model_file, bytes)
        self.net = self.ie_core.read_network(model=model_file, weights=weights_file, init_from_buffer=init_from_buffer)
        self.input_keys = list(self.net.input_info.keys())
        self.output_keys = list(self.net.outputs.keys())

    def load_model(self, model_file: Union[str, bytes], weights_file: Union[str, bytes, None]):
        """Loads an OpenVINO or ONNX model, overwrite this function if you need to reshape the network.

        Or retrieve additional information from the network after loading it.

        Args:
            model_file (Union[str, bytes]): Path to the model file or bytes with data from OpenVINO's .xml file.
            weights_file (weights_file: Union[str, bytes, None]): A .xml, .bin or .onnx file to be loaded. if a .xml
                or .bin is provided a file of the same name with the
                other extension is also expected
        """
        if self.net is None:
            self.read_model(model_file, weights_file)

        self.model = self.ie_core.load_network(
            network=self.net, device_name=self.device, num_requests=self.num_requests
        )


class AsyncOpenVINOTask:
    """This class runs asynchronous inference on a BaseOpenVinoInferencer.

    Using a BaseStreamer as input

    Args:
        streamer: A streamer that provides input for the inferencer
        inferencer: The inferencer to use to generate predictions
        drop_output: Set to a number to limit the amount of results
            stored at a time. If inference is completed but there is
            no room for the output. The output will be dropped.Set
            to 0 to disable, Set to None to automatically determine
            a good value
    """

    def __init__(
        self,
        streamer: BaseStreamer,
        inferencer: BaseOpenVINOInferencer,
        drop_output: Optional[int] = None,
    ):
        self.streamer: BaseStreamer = streamer
        self.inferencer: BaseOpenVINOInferencer = inferencer

        if drop_output is None:
            # Setting to 2x number of requests should allow for a good balance
            # between memory conservation and flexibility.
            drop_output = self.inferencer.num_requests * 2

        self.drop_output = drop_output

    def __iter__(self) -> Iterator[Tuple[np.ndarray, List[np.ndarray]]]:
        """Starts the asynchronous inference loop.

        Example:
            >>> streamer = VideoStreamer("../demo.mp4")
            >>> inferencer = ExampleOpenVINOInferencer(weights="model.bin", num_requests=4)
            >>> async_task = AsyncOpenVINOTask(streamer, inferencer)
            >>> for image, predictions in async_task:
            ...    # Do something with predictions

        Yields:
            Iterator[Tuple[np.ndarray, List[np.ndarray]]]: A Tuple with the used image and a list of predictions
        """
        manager = multiprocessing.Manager()
        completed_requests = manager.Queue(maxsize=self.drop_output)

        if len(self.inferencer.model.requests) == 1:
            warnings.warn(
                "Using AsyncOpenVINOTask while num_requests of the inferencer is set "
                "to 1, use a higher value to get a performance benefit or use "
                "synchronous execution."
            )

        try:
            for frame in self.streamer:
                while not self.__idle_request_available():
                    try:
                        yield completed_requests.get(timeout=0.1)
                    except queue.Empty:
                        pass
                self.__make_request(frame, completed_requests)

            while not completed_requests.empty():
                try:
                    yield completed_requests.get(timeout=0.1)
                except queue.Empty:
                    self.__wait_for_request()
        except GeneratorExit:
            pass

    def __idle_request_available(self) -> bool:
        """Returns True if one idle request is available.

        Returns:
            bool: True if one idle request is available
        """
        return self.inferencer.model.get_idle_request_id() >= 0

    def __wait_for_request(self, num_requests: Optional[int] = None, timeout: Optional[int] = None) -> bool:
        """Wait for num_requests to become available.

        Args:
            num_requests: Number of requests that should be available
                for the function to return False. If set to None waits
                for all requests to finish. Defaults to None.
            timeout: Amount of milliseconds to wait before function
                returns regardless of available requests. Set to None to
                wait regardless of the time. Defaults to None.

        Returns:
            bool -- Returns True if no requests are available, False if
            num_requests are available
        """
        return self.inferencer.model.wait(num_requests=num_requests, timeout=timeout) == RESULT_NOT_READY

    def __make_request(self, image: np.ndarray, completed_requests: queue.Queue):
        """Makes an asynchronous request.

        Args:
            image: Image to run inference on.
            completed_requests: Queue where results should be placed.

        Raises:
            RuntimeError: Raised if no idle requests are available
        """
        request_id = self.inferencer.model.get_idle_request_id()
        if request_id == -1:
            raise RuntimeError("Tried to get idle request but got no request")
        request = self.inferencer.model.requests[request_id]
        input_image, metadata = self.inferencer.pre_process(image)

        request.set_completion_callback(
            py_callback=_async_callback,
            py_data=(self, request, image, metadata, completed_requests),
        )
        request.async_infer(inputs=input_image)


def _async_callback(
    status,
    callback_args: Tuple[AsyncOpenVINOTask, InferRequest, np.ndarray, Any, multiprocessing.Queue],
):
    """Callback for Async Infer.

    Adds the used image and output Dictionary to the completed_requests Queue.

    Args:
        status: OpenVINO status code
        callback_args: AsyncOpenVINOTask object, the Inference Request,
            the image used, Queue to put the output in.
    """
    self, request, image, metadata, completed_requests = callback_args
    try:
        if status != OK:
            raise RuntimeError(f"Infer request has returned status code {status}")

        output_blobs = request.output_blobs
        output_blobs = {k: output_blob.buffer for (k, output_blob) in output_blobs.items()}
        output = self.inferencer.post_process(output_blobs, metadata)

        try:
            completed_requests.put_nowait((image, output))
        except queue.Full:
            logger.warning("An inference result was dropped because the queue is full")
            # TODO: Make the callback safely drop the oldest output and add the
            #  new output
        except AttributeError:
            # If __iter__ has exited while requests are still up this exception is
            # thrown
            pass

    except RuntimeError as error:
        logger.warning("RunTimeError in AsyncOpenVINOTask: _async_callback: %s", str(error))
