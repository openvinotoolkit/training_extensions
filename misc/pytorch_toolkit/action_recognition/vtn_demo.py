import sys
import time
from argparse import ArgumentParser
from collections import deque
from copy import deepcopy

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from action_recognition.model import create_model
from action_recognition.options import add_input_args
from action_recognition.spatial_transforms import (CenterCrop, Compose,
                                                   Normalize, Scale, ToTensor, MEAN_STATISTICS, STD_STATISTICS)
from action_recognition.utils import load_state, generate_args

TEXT_COLOR = (255, 255, 255)
TEXT_FONT_FACE = cv2.FONT_HERSHEY_DUPLEX
TEXT_FONT_SIZE = 1
TEXT_VERTICAL_INTERVAL = 45
NUM_LABELS_TO_DISPLAY = 2


class TorchActionRecognition:
    def __init__(self, encoder, checkpoint_path, num_classes=400, **kwargs):
        model_type = "{}_vtn".format(encoder)
        args, _ = generate_args(model=model_type, n_classes=num_classes, layer_norm=False, **kwargs)
        self.args = args
        self.model, _ = create_model(args, model_type)

        self.model = self.model.module
        self.model.eval()
        self.model.cuda()

        checkpoint = torch.load(str(checkpoint_path))
        load_state(self.model, checkpoint['state_dict'])

        self.preprocessing = make_preprocessing(args)
        self.embeds = deque(maxlen=(args.sample_duration * args.temporal_stride))

    def preprocess_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.preprocessing(frame)

    def infer_frame(self, frame):
        embedding = self._infer_embed(self.preprocess_frame(frame))
        self.embeds.append(embedding)
        sequence = self.get_seq()
        return self._infer_logits(sequence)

    def _infer_embed(self, frame):
        with torch.no_grad():
            frame_tensor = frame.unsqueeze(0).to('cuda')
            tensor = self.model.resnet(frame_tensor)
            tensor = self.model.reduce_conv(tensor)
            embed = F.avg_pool2d(tensor, 7)
        return embed.squeeze(-1).squeeze(-1)

    def _infer_logits(self, embeddings):
        with torch.no_grad():
            ys = self.model.self_attention_decoder(embeddings)
            ys = self.model.fc(ys)
            ys = ys.mean(1)
        return ys.cpu()

    def _infer_seq(self, frame):
        with torch.no_grad():
            result = self.model(frame.view(1, self.args.sample_duration, 3,
                                           self.args.sample_size, self.args.sample_size).to('cuda'))
        return result.cpu()

    def get_seq(self):
        sequence = torch.stack(tuple(self.embeds), 1)
        if self.args.temporal_stride > 1:
            sequence = sequence[:, ::self.args.temporal_stride, :]

        n = self.args.sample_duration
        if sequence.size(1) < n:
            num_repeats = (n - 1) // sequence.size(1) + 1
            sequence = sequence.repeat(1, num_repeats, 1)[:, :n, :]

        return sequence


def make_preprocessing(args):
    return Compose([
        Scale(args.sample_size),
        CenterCrop(args.sample_size),
        ToTensor(args.norm_value),
        Normalize(MEAN_STATISTICS[args.mean_dataset], STD_STATISTICS[args.mean_dataset])
    ])


def draw_rect(image, bottom_left, top_right, color=(0, 0, 0), alpha=1.):
    xmin, ymin = bottom_left
    xmax, ymax = top_right

    image[ymin:ymax, xmin:xmax, :] = image[ymin:ymax, xmin:xmax, :] * (1 - alpha) + np.asarray(color) * alpha
    return image


def render_frame(frame, probs, labels):
    order = probs.argsort(descending=True)

    status_bar_coordinates = (
        (0, 0),  # top left
        (650, 25 + TEXT_VERTICAL_INTERVAL * NUM_LABELS_TO_DISPLAY)  # bottom right
    )

    draw_rect(frame, status_bar_coordinates[0], status_bar_coordinates[1], alpha=0.5)

    for i, imax in enumerate(order[:NUM_LABELS_TO_DISPLAY]):
        text = '{} - {:.1f}%'.format(labels[imax], probs[imax] * 100)
        text = text.upper().replace("_", " ")
        cv2.putText(frame, text, (15, TEXT_VERTICAL_INTERVAL * (i + 1)), TEXT_FONT_SIZE,
                    TEXT_FONT_FACE, TEXT_COLOR)

    return frame


def run_demo(model, video_cap, labels):
    fps = float(video_cap.get(cv2.CAP_PROP_FPS))
    print('fps: {}'.format(fps))
    if not fps:
        fps = 30.0
    delay = max(1, int(1000 / fps))
    tick = time.time()

    while video_cap.isOpened():
        ok, frame = video_cap.read()

        if not ok:
            break

        logits = model.infer_frame(frame)
        probs = F.softmax(logits[0], dim=0)
        frame = render_frame(frame, probs, labels)

        tock = time.time()
        expected_time = tick + 1 / fps
        if tock < expected_time:
            delay = max(1, int((expected_time - tock) * 1000))
        tick = tock

        cv2.imshow("demo", frame)
        key = cv2.waitKey(delay)
        if key == 27 or key == ord('q'):
            break


def main():
    parser = ArgumentParser()
    parser.add_argument("--encoder", help="What encoder to use ", default='resnet34')
    parser.add_argument("--checkpoint", help="Path to pretrained model (.pth) file", required=True)
    parser.add_argument("--input-video", type=str, help="Path to input video", required=True)
    parser.add_argument("--labels", help="Path to labels file (new-line separated file with label names)", type=str,
                        required=True)
    add_input_args(parser)
    args = parser.parse_args()

    with open(args.labels) as fd:
        labels = fd.read().strip().split('\n')

    extra_args = deepcopy(vars(args))
    input_data_params = set(x.dest for x in parser._action_groups[-1]._group_actions)
    for name in list(extra_args.keys()):
        if name not in input_data_params:
            del extra_args[name]

    model = TorchActionRecognition(args.encoder, args.checkpoint, num_classes=len(labels), **extra_args)
    cap = cv2.VideoCapture(args.input_video)
    run_demo(model, cap, labels)


if __name__ == '__main__':
    sys.exit(main())
