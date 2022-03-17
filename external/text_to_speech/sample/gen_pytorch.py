import os
import time
from argparse import ArgumentParser, SUPPRESS

import librosa
import torch
import soundfile as sf
import numpy as np
from collections import OrderedDict

from torchtts.models import GANTacotron
from torchtts.utils import load_cfg
from torchtts.text_preprocessing import text_to_sequence, intersperse
from torchtts.text_preprocessing.symbols import symbols
from torchtts.text_preprocessing import cmudict


def denormalize_amp(S):
    return np.exp(S)


def reconstruct_waveform(mel, cfg, n_iter=32):
    """Uses Griffin-Lim phase reconstruction to convert from a normalized
    mel spectrogram back into a waveform."""
    amp_mel = denormalize_amp(mel)
    sample_rate = cfg.sampling_rate
    n_fft = cfg.filter_length
    hop_length = cfg.hop_length
    win_length = cfg.win_length
    fmin = cfg.mel_fmin
    fmax = cfg.mel_fmax

    S = librosa.feature.inverse.mel_to_stft(
        amp_mel, power=1, sr=sample_rate,
        n_fft=n_fft, fmin=fmin, fmax=fmax)
    wav = librosa.core.griffinlim(
        S, n_iter=n_iter,
        hop_length=hop_length, win_length=win_length)
    return wav


def save_wav(x, path, sample_rate = 22050):
    sf.write(path, x, sample_rate, 'PCM_24')


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')

    parser.add_argument('--model_dir', type=str, help='[string/path] Load weights')
    parser.add_argument('--cfg', type=str, default=None, help='[string/path] ')
    parser.add_argument('--onnx_dir', type=str, default=None, help='[string/path] ')
    parser.add_argument('--file', type=str, default=None, help='[string/path] ')

    args.add_argument("-o", "--out", help="Required. Path to an output directory", required=True,
                      type=str)

    return parser


def main():
    args = build_argparser().parse_args()

    cfg = load_cfg(args.cfg)

    device = torch.device('cpu')
    print('Using device:', device)

    # Instantiate TTS Model
    print('\nInitialising TTS Model...\n')
    cfg.model.encoder.num_chars = len(symbols)
    model = GANTacotron(cfg.model).to(device)

    state_dict = torch.load(args.model_dir, map_location=torch.device('cpu'))

    new_state = OrderedDict()

    if 'epoch' in state_dict:
        print('Epoch: ', state_dict['epoch'])
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    else:
        state_dict = state_dict['model']

    for k, v in state_dict.items():
        if 'discriminator.' in k:
            continue
        if 'generator.' in k:
            new_k = k[10:]
        else:
            new_k = k
        new_state[new_k] = v

    model.load_state_dict(new_state)
    _ = model.eval()

    with torch.no_grad():
        model.remove_weight_norm()

    cmudict_loader = None
    if not (getattr(cfg.data, "cmudict_path", None)) is dict:
        cmudict_loader = cmudict.CMUDict('../' + cfg.data.cmudict_path)

    if args.onnx_dir is not None:
        if not os.path.exists(args.onnx_dir):
            os.mkdir(args.onnx_dir)
        model.to_onnx(args.onnx_dir, cfg)

    input_text = [
        "Hello! My name is Martin Kronberg. And this is the IOT developer show, season two. During our break, we've been busy reworking the show. So think of this less like a sequel, and more like a greedy reboot."]

    if args.file is not None:
        input_text = []
        with open(args.file, 'r') as f:
            for line in f:
                input_text.append(line.lstrip())

    out_dir = args.out
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for i, tts_in in enumerate(input_text):
        text_norm = text_to_sequence(tts_in.rstrip(), ['english_cleaners'], cmudict_loader)

        if not type(getattr(cfg.data, "add_blank")) is dict:
            text_norm = intersperse(text_norm)

        sequence = np.array(text_norm)[None, :]
        x_tst = torch.from_numpy(sequence).to(device).long()
        x_tst_lengths = torch.tensor([x_tst.shape[1]]).to(device)

        with torch.no_grad():
            start_t = time.perf_counter()
            y_gen_tst, x_m, mel_lengths = model.generate(x_tst, x_tst_lengths)

        y_gen_tst = y_gen_tst.squeeze(0).cpu().detach().numpy()
        y_gen_tst = y_gen_tst * 6.0 - 6.0

        wav = reconstruct_waveform(y_gen_tst, cfg.data, 64)
        out_name = os.path.join(out_dir, "audio{0}.wav".format(i))
        print(out_name)
        save_wav(wav, out_name)
        np.save(os.path.join(out_dir, "mel{0}.npy".format(i)), y_gen_tst)
        end_t = time.perf_counter()
        print("Time: {0}. Text len: {1}.".format(end_t - start_t, len(tts_in)))


if __name__ == "__main__":
    main()


