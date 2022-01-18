# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import logging
import os
import tempfile

import inspect
import json

from shutil import copyfile, copytree
from zipfile import ZipFile
from typing import Any, Dict, Tuple, Optional, Union
import numpy as np

import ote_sdk.usecases.exportable_code.demo as demo
from ote_sdk.entities.inference_parameters import InferenceParameters, default_progress_callback
from ote_sdk.usecases.evaluation.metrics_helper import MetricsHelper
from ote_sdk.usecases.exportable_code.inference import BaseOpenVINOInferencer
from ote_sdk.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from ote_sdk.usecases.tasks.interfaces.inference_interface import IInferenceTask
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.model import (
    ModelEntity,
)

from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.annotation import AnnotationSceneEntity
from ote_sdk.entities.datasets import DatasetEntity

from torchtts.integration.parameters import OTETextToSpeechTaskParameters
from torchtts.datasets import text_to_sequence, intersperse, pad_spaces

logger = logging.getLogger(__name__)


def get_output(net, outputs, name):
    try:
        key = net.get_ov_name_for_tensor(name)
        assert key in outputs, f'"{key}" is not a valid output identifier'
    except KeyError as err:
        if name not in outputs:
            raise KeyError(f'Failed to identify output "{name}"') from err
        key = name
    return outputs[key]


class OpenVINOTTSInferencer(BaseOpenVINOInferencer):
    def __init__(
        self,
        model_file: Union[str, bytes],
        weight_file: Union[str, bytes, None] = None,
        device: str = "CPU",
        num_requests: int = 1,
    ):
        """
        Inferencer implementation for OTEDetection using OpenVINO backend.
        :param model: Path to model to load, `.xml`, `.bin` or `.onnx` file.
        :param hparams: Hyper parameters that the model should use.
        :param num_requests: Maximum number of requests that the inferencer can make.
            Good value is the number of available cores. Defaults to 1.
        :param device: Device to run inference on, such as CPU, GPU or MYRIAD. Defaults to "CPU".
        """
        super().__init__(model_file, weight_file, device, num_requests)

    def load_model(
        self, model_file: Union[str, bytes], weights_file: Union[str, bytes, None]
    ):
        self.net = self.ie_core.read_network(model_file, weights_file)

        scales = 9
        t_shapes = [int(128 * 1.5 ** i) for i in range(scales)]

        orig_shapes = {k: self.net.input_info[k].input_data.shape for k in self.net.input_info.keys()}
        exec_net = []
        for s in range(scales):
            new_shapes = {k: tuple(list(v[:-1]) + [t_shapes[s]]) if len(v) > 1 else v
                          for k, v in orig_shapes.items()}
            self.net.reshape(new_shapes)
            exec_net.append(
                self.ie_core.load_network(network=self.net, device_name=self.device, num_requests=self.num_requests))
            self.net.reshape(orig_shapes)

        self.model = exec_net
        self.t_shapes = t_shapes

    def post_process(self, prediction: Any, metadata: Any) -> AnnotationSceneEntity:
        return prediction


class OpenVINOTTSEncoderInferencer(OpenVINOTTSInferencer):
    def __init__(
        self,
        model_file: Union[str, bytes],
        weight_file: Union[str, bytes, None] = None,
    ):
        super().__init__(model_file, weight_file)

        self.load_model(model_file, weight_file)

        self.input_data_name = "seq"
        self.input_mask_name = "seq_len"

    def pre_process(self, text: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        x = text_to_sequence(text, self.cfg.text_cleaners, self.cmudict)
        if getattr(self.cfg, "add_blank", False):
            x = intersperse(x)

        for t_shape in self.t_shapes:
            if t_shape >= x.shape[-1]:
                seq = pad_spaces(x, t_shape)
                break

        seq = np.array(x)[None, :]
        seq_len = np.array([seq.shape[1]])

        dict_inputs = {self.input_data_name: seq, self.input_mask_name: seq_len}
        return dict_inputs

    def pre_process(self, seq: np.ndarray, seq_len: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        for t_shape in self.t_shapes:
            if t_shape >= seq.shape[-1]:
                seq = pad_spaces(seq, t_shape)
                break
        dict_inputs = {self.input_data_name: seq, self.input_mask_name: seq_len}
        return dict_inputs

    def forward(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for i in range(len(self.t_shapes)):
            if data[self.input_data_name].shape[-1] == self.t_shapes[i]:
                return self.model[i].infer(data)
        return None


class OpenVINOTTSDecoderInferencer(OpenVINOTTSInferencer):
    def __init__(
        self,
        model_file: Union[str, bytes],
        weight_file: Union[str, bytes, None] = None,
    ):
        super().__init__(model_file, weight_file)

        self.load_model(model_file, weight_file)

        self.input_data_name = "z"
        self.input_mask_name = "z_mask"

    @staticmethod
    def pad_last_dim(x, sz, constant_value=-1.0):
        pad_width = [(0, 0) for _ in range(len(x.shape) - 1)]
        pad_width.append((0, sz - x.shape[-1]))
        return np.pad(x, pad_width=pad_width, mode='constant', constant_values=constant_value)

    def pre_process(self, data: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        z = data[self.input_data_name]
        z_mask = data[self.input_mask_name]
        for t_shape in self.t_shapes:
            if t_shape >= z.shape[-1]:
                z = self.pad_last_dim(z, t_shape)
                z_mask = self.pad_last_dim(z_mask, t_shape, 0)
                break

        dict_inputs = {self.input_data_name: z, self.input_mask_name: z_mask}
        return dict_inputs

    def pre_process(self, z: np.ndarray, z_mask: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        for t_shape in self.t_shapes:
            if t_shape >= z.shape[-1]:
                z = self.pad_last_dim(z, t_shape)
                z_mask = self.pad_last_dim(z_mask, t_shape, 0)
                break

        dict_inputs = {self.input_data_name: z, self.input_mask_name: z_mask}
        return dict_inputs

    def forward(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for i in range(len(self.t_shapes)):
            if data[self.input_data_name].shape[-1] == self.t_shapes[i]:
                return self.model[i].infer(data)
        return None


class OpenVINOTTSTask(IInferenceTask, IEvaluationTask):
    def __init__(self, task_environment: TaskEnvironment):
        self.task_environment = task_environment
        self.hparams = self.task_environment.get_hyper_parameters(OTETextToSpeechTaskParameters)
        self.model = self.task_environment.model
        self.encoder, self.decoder = self.load_inferencers()

    def load_inferencers(self) -> Tuple[OpenVINOTTSInferencer]:
        encoder = OpenVINOTTSEncoderInferencer(self.model.get_data("encoder.xml"),
                                     self.model.get_data("encoder.bin"))
        decoder = OpenVINOTTSDecoderInferencer(self.model.get_data("decoder.xml"),
                                    self.model.get_data("decoder.bin"))
        return encoder, decoder

    @staticmethod
    def compute_train_attention_map(x_m: np.ndarray, mel: np.ndarray, attn_mask: np.ndarray) -> np.ndarray:
        mul = np.ones_like(x_m)  # [b, d, t]
        x_2 = np.sum(-x_m ** 2, 1)  # np.expand_dims(np.sum(-x_m ** 2, 1), axis=-1)  # [b, t, 1]
        z_2 = np.matmul(mul.transpose((0, 2, 1)), -mel ** 2)  # [b, t, t']
        xz2 = np.matmul(x_m.transpose((0, 2, 1)), mel)  # [b, t, d] * [b, d, t'] = [b, t, t']

        corr_coeff = z_2 + x_2[:, :, np.newaxis]
        corr_coeff = corr_coeff + 2 * xz2

        attn = OpenVINOTTSTask.maximum_path_np(corr_coeff, np.squeeze(attn_mask, 1)).astype(x_m.dtype)
        return attn

    @staticmethod
    def maximum_path_np(map: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        map: [b, t_text, t_mel]
        mask: [b, t_text, t_mel]
        """
        map = map * mask

        path = np.zeros_like(map).astype(np.int32)

        t_text_max = mask.sum(1)[:, 0].astype(np.int32)
        t_mel_max = mask.sum(2)[:, 0].astype(np.int32)

        for b in range(map.shape[0]):
            min_val = -1e9
            t_text, t_mel = t_text_max[b], t_mel_max[b]
            index = t_text - 1

            for x in range(t_mel):
                for y in range(max(0, t_text + x - t_mel), min(t_text, x + 1)):
                    if x == y:
                        v_cur = min_val
                    else:
                        v_cur = map[b, y, x - 1]
                    if y == 0:
                        if x == 0:
                            v_prev = 0.
                        else:
                            v_prev = min_val
                    else:
                        v_prev = map[b, y - 1, x - 1]
                    map[b, y, x] = max(v_cur, v_prev) + map[b, y, x]

            for x in range(t_mel - 1, -1, -1):
                path[b, index, x] = 1
                if index != 0 and (index == x or map[b, index, x - 1] < map[b, index - 1, x - 1]):
                    index = index - 1

        return path

    @staticmethod
    def sequence_mask(length, max_length=None):
        if max_length is None:
            max_length = length.max()
        x = np.arange(max_length)
        x = np.expand_dims(x, 0)
        length = np.expand_dims(length, 1)
        res = x < length
        return res

    @staticmethod
    def generate_path(duration, mask):
        b, t_x, t_y = mask.shape  # batch size, text size, mel size
        cum_duration = np.cumsum(duration, 1)

        cum_duration_flat = cum_duration.flatten()  # view(b * t_x)
        path = OpenVINOTTSTask.sequence_mask(cum_duration_flat, t_y).astype(mask.dtype)

        path = path.reshape(b, t_x, t_y)
        path = path - np.pad(path, ((0, 0), (1, 0), (0, 0)))[:, :-1]

        path = path * mask

        return path

    @staticmethod
    def alignment(encoder_out, add_duration=0.1):
        x_res, log_dur, x_mask = encoder_out['x_res'], encoder_out['log_dur'], encoder_out['x_mask']

        w = (np.exp(log_dur) + add_duration) * x_mask
        w_ceil = np.ceil(w)

        mel_lengths = np.clip(np.sum(w_ceil, axis=(1, 2)), a_min=1, a_max=None).astype(dtype=np.long)

        z_mask = np.expand_dims(OpenVINOTTSTask.sequence_mask(mel_lengths), 1).astype(x_mask.dtype)
        attn_mask = np.expand_dims(x_mask, -1) * np.expand_dims(z_mask, 2)

        attn = OpenVINOTTSTask.generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1))
        attn = np.expand_dims(attn, 1)
        attn = attn.squeeze(1).transpose(0, 2, 1)
        z_m = np.matmul(attn, x_res.transpose(0, 2, 1)).transpose(0, 2, 1)  # [b, t', t], [b, t, d] -> [b, d, t']

        return {'z': z_m, 'z_mask': z_mask}

    @staticmethod
    def sequence_mask(length, max_length=None):
        if max_length is None:
            max_length = length.max()
        x = np.arange(max_length)
        x = np.expand_dims(x, 0)
        length = np.expand_dims(length, 1)
        res = x < length
        return res

    def infer_like_test(self, dataset_item):
        encoder_in = self.encoder.pre_process(dataset_item.numpy)
        encoder_out = self.encoder.forward(encoder_in)

        decoder_in = self.decoder.pre_process(self.alignment(encoder_out))
        mel_spectrogram = self.decoder.forward(decoder_in)
        return mel_spectrogram

    def infer_like_train(self, dataset_item):
        x, x_len, mel, mel_len = dataset_item

        res_mel = []
        res_attn = []

        for bs in range(x.shape[0]):
            encoder_in = self.encoder.pre_process(x[[bs], :], x_len[[bs]])
            encoder_out = self.encoder.forward(encoder_in)

            x_m, x_res, logw, x_mask = encoder_out["x_m"], encoder_out["x_logs"], \
                                       encoder_out["logw"], encoder_out["x_mask"]

            mel_max_length = mel.shape[2]
            z_mask = self.sequence_mask(mel_len[[bs]], mel_max_length).astype(x_m.dtype)
            z_mask = np.expand_dims(z_mask, axis=[1, 2])
            x_mask = np.expand_dims(x_mask, axis=[-1])
            attn_mask = x_mask * z_mask
            z_mask = np.squeeze(z_mask, 2)

            attn = self.compute_train_attention_map(x_m, mel[[bs], :, :], attn_mask)

            z_m = np.matmul(attn.transpose((0, 2, 1)), x_res.transpose((0, 2, 1))).transpose((0, 2, 1))

            decoder_in = self.decoder.pre_process(z_m, z_mask)
            mel_out = self.decoder.forward(decoder_in)['mel']

            res_mel.append(mel_out)
            res_attn.append(attn)
        mel = np.concatenate(res_mel, axis=0)
        attn = np.concatenate(res_attn, axis=0)

        return mel, attn

    def infer(self, dataset: DatasetEntity,
              inference_parameters: Optional[InferenceParameters] = None) -> DatasetEntity:
        return dataset

    def evaluate(self,
                 output_result_set: ResultSetEntity,
                 evaluation_metric: Optional[str] = None):
        if evaluation_metric is not None:
            logger.warning(f'Requested to use {evaluation_metric} metric,'
                            'but parameter is ignored. Use accuracy instead.')
        output_result_set.performance = MetricsHelper.compute_accuracy(output_result_set).get_performance()

    def deploy(self, output_model: ModelEntity) -> None:
        """Exports the weights from ``output_model`` along with exportable code.

        Args:
            output_model (ModelEntity): Model with ``openvino.xml`` and ``.bin`` keys

        Raises:
            Exception: If ``task_environment.model`` is None
        """
        logger.info("Deploying Model")

        if self.task_environment.model is None:
            raise Exception("task_environment.model is None. Cannot load weights.")

        work_dir = os.path.dirname(demo.__file__)
        parameters: Dict[str, Any] = {}
        parameters["type_of_model"] = "text_to_speech"
        parameters["converter_type"] = "TEXT_TO_SPEECH"
        parameters["model_parameters"] = {}
        name_of_package = "demo_package"

        with tempfile.TemporaryDirectory() as tempdir:
            copyfile(os.path.join(work_dir, "setup.py"), os.path.join(tempdir, "setup.py"))
            copyfile(os.path.join(work_dir, "requirements.txt"), os.path.join(tempdir, "requirements.txt"))
            copytree(os.path.join(work_dir, name_of_package), os.path.join(tempdir, name_of_package))
            config_path = os.path.join(tempdir, name_of_package, "config.json")
            with open(config_path, "w", encoding="utf-8") as file:
                json.dump(parameters, file, ensure_ascii=False, indent=4)

            copyfile(inspect.getfile(AnomalyClassification), os.path.join(tempdir, name_of_package, "model.py"))

            # create wheel package
            subprocess.run(
                [
                    sys.executable,
                    os.path.join(tempdir, "setup.py"),
                    "bdist_wheel",
                    "--dist-dir",
                    tempdir,
                    "clean",
                    "--all",
                ],
                check=True,
            )
            wheel_file_name = [f for f in os.listdir(tempdir) if f.endswith(".whl")][0]

            with ZipFile(os.path.join(tempdir, "openvino.zip"), "w") as arch:
                arch.writestr(os.path.join("model", "model.xml"), self.task_environment.model.get_data("openvino.xml"))
                arch.writestr(os.path.join("model", "model.bin"), self.task_environment.model.get_data("openvino.bin"))
                arch.write(os.path.join(tempdir, "requirements.txt"), os.path.join("python", "requirements.txt"))
                arch.write(os.path.join(work_dir, "README.md"), os.path.join("python", "README.md"))
                arch.write(os.path.join(work_dir, "demo.py"), os.path.join("python", "demo.py"))
                arch.write(os.path.join(tempdir, wheel_file_name), os.path.join("python", wheel_file_name))
            with open(os.path.join(tempdir, "openvino.zip"), "rb") as output_arch:
                output_model.exportable_code = output_arch.read()
        logger.info("Deploying completed")