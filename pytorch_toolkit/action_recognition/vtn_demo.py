import sys
import time
from argparse import ArgumentParser
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from action_recognition.model import create_model
from action_recognition.spatial_transforms import (CenterCrop, Compose,
                                                   Normalize, Scale, ToTensor, MEAN_STATISTICS, STD_STATISTICS)
from action_recognition.utils import load_state, generate_args

TEXT_COLOR = (255, 255, 255)
TEXT_FONT_FACE = cv2.FONT_HERSHEY_DUPLEX
TEXT_FONT_SIZE = 1
TEXT_VERTICAL_INTERVAL = 45
NUM_LABELS_TO_DISPLAY = 2


class TorchActionRecognition:
    def __init__(self, encoder, checkpoint_path, num_classes=400):
        model_type = "{}_vtn".format(encoder)
        args, _ = generate_args(model=model_type, n_classes=num_classes, layer_norm=False)
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
        embedding = self._infer_embed(frame)
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
            result = self.model(frame.view(1, 16, 3, 224, 224).to('cuda'))
        return result.cpu()

    def get_seq(self):
        sequence = torch.stack(tuple(self.embeds), 1)
        sequence = sequence[:, ::2, :]

        if sequence.size(1) < 16:
            num_repeats = 15 // sequence.size(1) + 1
            sequence = sequence.repeat(1, num_repeats, 1)[:, :16, :]

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

    status_bar_coorinates = (
        (0, 0),  # top left
        (650, 25 + TEXT_VERTICAL_INTERVAL * NUM_LABELS_TO_DISPLAY)  # bottom right
    )

    draw_rect(frame, status_bar_coorinates[0], status_bar_coorinates[1], alpha=0.5)

    for i, imax in enumerate(order[:NUM_LABELS_TO_DISPLAY]):
        text = '{} - {:.1f}%'.format(labels[imax], probs[imax] * 100)
        text = text.upper().replace("_", " ")
        cv2.putText(frame, text, (15, TEXT_VERTICAL_INTERVAL * (i + 1)), TEXT_FONT_SIZE,
                    TEXT_FONT_FACE, TEXT_COLOR)

    return frame


def run_demo(model, video_cap, labels):
    tick = time.time()
    while video_cap.isOpened():
        ok, frame = video_cap.read()

        if not ok:
            break

        logits = model.infer_frame(model.preprocess_frame(frame))
        probs = F.softmax(logits[0])
        frame = render_frame(frame, probs, labels)

        tock = time.time()
        expected_time = tick + 1 / 30.
        if tock < expected_time:
            time.sleep(expected_time - tock)
        tick = tock

        cv2.imshow("demo", frame)
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break


def main():
    parser = ArgumentParser()
    parser.add_argument("--encoder", help="What encoder to use ", default='resnet34')
    parser.add_argument("--checkpoint", help="Path to pretrained model (.pth) file", required=True)
    parser.add_argument("--input-video", type=str, help="Path to input video", required=True)
    parser.add_argument("--labels", help="Path to labels file (new-line separated file with label names)", type=str,
                        required=True)
    args = parser.parse_args()

    with open(args.labels) as fd:
        labels = fd.read().strip().split('\n')

    model = TorchActionRecognition(args.encoder, args.checkpoint, num_classes=len(labels))
    cap = cv2.VideoCapture(args.input_video)
    run_demo(model, cap, labels)


if __name__ == '__main__':
    sys.exit(main())
