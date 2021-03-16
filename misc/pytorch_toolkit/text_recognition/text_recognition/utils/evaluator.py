"""
 Copyright (c) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import json
import os.path
from copy import deepcopy
from enum import Enum
from functools import partial

import numpy as np
import onnxruntime
import torch
from openvino.inference_engine import IECore
from scipy.special import log_softmax
from torch.utils.data import DataLoader
from tqdm import tqdm
from text_recognition.data.utils import collate_fn, create_list_of_transforms, ctc_greedy_search
from text_recognition.data.vocab import END_TOKEN, START_TOKEN, read_vocab
from text_recognition.datasets.dataset import str_to_class
from text_recognition.models.model import TextRecognitionModel
from text_recognition.utils.common import DECODER_INPUTS, DECODER_OUTPUTS, ENCODER_INPUTS, ENCODER_OUTPUTS, read_net
from text_recognition.utils.evaluation_utils import Im2latexRenderBasedMetric

MAX_SEQ_LEN = 256

spaces = [r'\,', r'\>', r'\;', r'\:', r'\quad', r'\qquad', '~']


def ends_with_space(string):
    """If string end with one of the latex spaces (given the above),
    returns True and index of this space, else False and None

    Args:
        string (str): input string with possible spaces

    Returns:
        Tuple(bool, int) string ends with space, index of the space
    """
    for idx, space in enumerate(spaces):
        if string.endswith(space):
            return True, idx
    return False, None


def get_onnx_inputs(model):
    names = []
    for inp in model.get_inputs():
        names.append(inp.name)
    return names


def get_onnx_outputs(model):
    names = []
    for out in model.get_outputs():
        names.append(out.name)
    return names


def postprocess_prediction(pred_phrase_str):
    """Deletes usual space in the end of the string and then checks
    if string ends with latex space. If yes, deletes latex space.
    Deletion of spaces is performed because, even though spaces in the end are invisible,
    they affect on rendering the formula, making it more tight to the left

    Args:
        pred_phrase_str (str): input string

    Returns:
        str: postprocessed string
    """
    pred_phrase_str = pred_phrase_str.rstrip()
    ends, idx = ends_with_space(pred_phrase_str)
    while ends:
        pred_phrase_str = pred_phrase_str[:len(pred_phrase_str) - len(spaces[idx])]
        pred_phrase_str = pred_phrase_str.rstrip()
        ends, idx = ends_with_space(pred_phrase_str)
    return pred_phrase_str


class RunnerType(Enum):
    PyTorch = 0
    ONNX = 1
    OpenVINO = 2


class BaseRunner:
    def __init__(self, config):
        self.config = config

    def load_model(self):
        raise NotImplementedError

    def run_model(self, img):
        raise NotImplementedError

    def openvino_transform(self):
        raise NotImplementedError


class PyTorchRunner(BaseRunner):
    def load_model(self):
        self.vocab_len = len(read_vocab(self.config.get('vocab_path')))
        self.use_ctc = self.config.get('use_ctc')
        out_size = self.vocab_len + 1 if self.use_ctc else self.vocab_len
        self.model = TextRecognitionModel(self.config.get('backbone_config'), out_size, self.config.get('head', {}))
        self.device = self.config.get('device', 'cpu')
        self.model.load_weights(self.config.get('model_path'), map_location=self.device)
        self.model = self.model.to(self.device)
        self.model.eval()

    def run_model(self, img):
        img = img.to(self.device)
        logits, pred = self.model(img)
        if self.use_ctc:
            pred = torch.nn.functional.log_softmax(logits.detach(), dim=2)
            pred = ctc_greedy_search(pred, 0)
        return pred[0]

    def openvino_transform(self):
        return False

    def reload_model(self, new_model_path):
        self.model.load_weights(new_model_path, map_location=self.device)
        self.model = self.model.to(self.device)
        self.model.eval()


class ONNXRunner(BaseRunner):
    def load_model(self):
        self.use_ctc = self.config.get('use_ctc')
        if self.use_ctc:
            self.model = onnxruntime.InferenceSession(self.config.get('res_model_name'))
        else:
            self.decoder_onnx = onnxruntime.InferenceSession(self.config.get('res_decoder_name'))
            self.encoder_onnx = onnxruntime.InferenceSession(self.config.get('res_encoder_name'))

    def run_decoder(self, hidden, context, output, row_enc_out):

        decoder_inputs = get_onnx_inputs(self.decoder_onnx)
        decoder_outputs = get_onnx_outputs(self.decoder_onnx)
        logits = []
        logit = None
        for _ in range(MAX_SEQ_LEN):
            if logit is not None:
                tgt = np.reshape(np.argmax(logit, axis=1), (1, 1)).astype(np.long)
            else:
                tgt = np.array([[START_TOKEN]] * 1)
            if tgt[0][0] == END_TOKEN:
                break
            hidden, context, output, logit = self.decoder_onnx.run(
                decoder_outputs,
                {
                    decoder_inputs[0]: hidden,
                    decoder_inputs[1]: context,
                    decoder_inputs[2]: output,
                    decoder_inputs[3]: row_enc_out,
                    decoder_inputs[4]: tgt
                })
            logits.append(logit)
        return np.argmax(np.array(logits).squeeze(1), axis=1)

    def run_encoder(self, img):
        encoder_outputs = get_onnx_outputs(self.encoder_onnx)
        encoder_input = get_onnx_inputs(self.encoder_onnx)[0]
        return self.encoder_onnx.run(encoder_outputs, {
            encoder_input: np.array(img, dtype=np.float32)
        })

    def run_complete_model(self, img):
        model_output_names = get_onnx_outputs(self.model)
        model_input_names = get_onnx_inputs(self.model)[0]
        logits, _ = self.model.run(model_output_names, {
            model_input_names: np.array(img, dtype=np.float32)
        })
        pred = log_softmax(logits, axis=2)
        pred = ctc_greedy_search(pred, 0)
        return pred[0]

    def run_model(self, img):
        img = img.clone().detach().numpy()
        if self.use_ctc:
            return self.run_complete_model(img)
        else:
            row_enc_out, h, c, O_t = self.run_encoder(img)
            pred = self.run_decoder(h, c, O_t, row_enc_out).astype(np.int32)
        return pred

    def openvino_transform(self):
        return False


class OpenVINORunner(BaseRunner):
    def load_model(self):
        ie = IECore()
        self.use_ctc = self.config.get('use_ctc')
        if self.use_ctc:
            model = read_net(self.config.get('res_model_name').replace('.onnx', '.xml'), ie)
            self.exec_net = ie.load_network(network=model, device_name='CPU')
        else:
            encoder = read_net(self.config.get('res_encoder_name').replace('.onnx', '.xml'), ie)
            dec_step = read_net(self.config.get('res_decoder_name').replace('.onnx', '.xml'), ie)
            self.exec_net_encoder = ie.load_network(network=encoder, device_name='CPU')
            self.exec_net_decoder = ie.load_network(network=dec_step, device_name='CPU')

    def run_model(self, img):
        if torch.is_tensor(img):
            img = img.clone().detach().numpy()
        if self.use_ctc:
            logits = self.exec_net.infer(inputs={self.config.get('model_input_names'): img})[
                self.config.get('model_output_names').split(',')[0]]
            pred = log_softmax(logits, axis=2)
            pred = ctc_greedy_search(pred, 0)
            return pred[0]
        else:
            enc_res = self.exec_net_encoder.infer(inputs={self.config.get(
                'encoder_input_names', ENCODER_INPUTS).split(',')[0]: img})
            enc_out_names = self.config.get('encoder_output_names', ENCODER_OUTPUTS).split(',')
            ir_row_enc_out = enc_res[enc_out_names[0]]
            dec_states_h = enc_res[enc_out_names[1]]
            dec_states_c = enc_res[enc_out_names[2]]
            output = enc_res[enc_out_names[3]]
            dec_in_names = self.config.get('decoder_input_names', DECODER_INPUTS).split(',')
            dec_out_names = self.config.get('decoder_output_names', DECODER_OUTPUTS).split(',')
            tgt = np.array([[START_TOKEN]] * 1)
            logits = []
            for _ in range(MAX_SEQ_LEN):
                dec_res = self.exec_net_decoder.infer(inputs={
                    dec_in_names[0]: dec_states_h,
                    dec_in_names[1]: dec_states_c,
                    dec_in_names[2]: output,
                    dec_in_names[3]: ir_row_enc_out,
                    dec_in_names[4]: tgt
                }
                )

                dec_states_h = dec_res[dec_out_names[0]]
                dec_states_c = dec_res[dec_out_names[1]]
                output = dec_res[dec_out_names[2]]
                logit = dec_res[dec_out_names[3]]
                logits.append(logit)

                tgt = np.reshape(np.argmax(logit, axis=1), (1, 1)).astype(np.long)
                if tgt[0][0] == END_TOKEN:
                    break
            return np.argmax(np.array(logits).squeeze(1), axis=1)

    def openvino_transform(self):
        return True


def create_runner(config, runner_type):
    if runner_type == RunnerType.PyTorch:
        return PyTorchRunner(config)
    if runner_type == RunnerType.ONNX:
        return ONNXRunner(config)
    if runner_type == RunnerType.OpenVINO:
        return OpenVINORunner(config)
    raise ValueError(f'Wrong type of the runner {runner_type}')


class Evaluator:
    def __init__(self, config, runner_type=RunnerType.PyTorch):
        self.config = deepcopy(config)
        self.runner = create_runner(self.config, runner_type)
        self.vocab = read_vocab(self.config.get('vocab_path'))
        self.render = self.config.get('render')
        self.load_dataset()
        self.runner.load_model()
        self.read_expected_outputs()

    def load_dataset(self):
        dataset_params = self.config.get('dataset')
        dataset_type = dataset_params.pop('type')

        val_dataset = str_to_class[dataset_type](**dataset_params)
        batch_transform = create_list_of_transforms(self.config.get(
            'val_transforms_list'), ovino_ir=self.runner.openvino_transform())
        print('Creating eval transforms list: {}'.format(batch_transform))
        self.val_loader = DataLoader(
            val_dataset,
            collate_fn=partial(collate_fn, self.vocab.sign2id,
                               batch_transform=batch_transform,
                               use_ctc=(self.config.get('use_ctc'))),
            num_workers=os.cpu_count())

    def read_expected_outputs(self):
        if self.config.get('expected_outputs'):
            with open(self.config.get('expected_outputs')) as outputs_file:
                self.expected_outputs = json.load(outputs_file)

    def validate(self):
        annotations = []
        predictions = []
        print('Starting inference')
        metric = Im2latexRenderBasedMetric()
        text_acc = 0
        for img_name, _, imgs, _, loss_computation_gt in tqdm(self.val_loader):
            with torch.no_grad():
                targets = self.runner.run_model(imgs)
                gold_phrase_str = self.vocab.construct_phrase(
                    loss_computation_gt[0], ignore_end_token=self.config.get('use_ctc'))
                pred_phrase_str = postprocess_prediction(self.vocab.construct_phrase(
                    targets, ignore_end_token=self.config.get('use_ctc')))
                annotations.append((gold_phrase_str, img_name[0]))
                predictions.append((pred_phrase_str, img_name[0]))
                text_acc += int(pred_phrase_str == gold_phrase_str)
        text_acc /= len(self.val_loader)
        print('Text accuracy is: ', text_acc)
        if not self.render:
            return text_acc
        res = metric.evaluate(annotations, predictions)
        return res
