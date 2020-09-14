import sys
import argparse

import numpy as np
import torch
import torch.onnx
import onnx
import onnxruntime as rt
import os.path
from functools import partial
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Compose
from utils import BatchTransformRescale, BatchToTensor

from data import Im2LatexDataset, BatchRandomSampler
from make_vocab import read_vocab, START_TOKEN
from model import Im2LatexModel
from utils import collate_fn, get_timestamp

from model import DEBUG_MO_1, DEBUG_ONNX


def main(args):
    parser = argparse.ArgumentParser(
        description="Im2Latex converting model to onnx program")

    # model args
    parser.add_argument(
        "--emb_dim", type=int, default=80, help="Embedding size")
    parser.add_argument(
        "--enc_rnn_h",
        type=int,
        default=256,
        help="The hidden state of the encoder RNN")
    parser.add_argument(
        "--dec_rnn_h",
        type=int,
        default=512,
        help="The hidden state of the decoder RNN")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./ckpt/best_model_epoch_2_2019-09-13_17-54-55_2_stages",
        help="The dataset's dir")
    parser.add_argument(
        "--data_path",
        type=str,
        default="../full_data/",
        help="The dataset's dir")
    parser.add_argument(
        "--res_encoder_name",
        type=str,
        default="encoder.onnx",
        help="Result encoder file")
    parser.add_argument(
        "--res_decoder_name",
        type=str,
        default="decoder.onnx",
        help="Result decoder file")
    parser.add_argument(
        "--cnn_encoder",
        type=str,
        default="our",
        choices=['our', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
                 'resnet152', 'resnext101_32x8d', 'resnext50_32x4d'],
        help="Name of the backbone")
    parser.add_argument(
        "--disable_layer_3", action='store_true', default=False, help="Disables layer3 in resnet")
    parser.add_argument(
        "--disable_layer_4", action='store_true', default=False, help="Disables layer4 in resnet")
    parser.add_argument(
        "--disable_last_conv", action='store_true', default=False, help="Disables last conv in resnet")
    parser.add_argument("--in_lstm_ch", type=int, default=512)
    parser.add_argument("--inp_channels", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_formula_len", type=int, default=256)
    parser.add_argument("--beam_width", type=int, default=0)
    parser.add_argument("--dataset", type=str, default='toy',
                        help='Defines which dataset to use')
    parser.add_argument(
        "--vocab_path",
        type=str,
        default="../full_data/vocab.pkl",
        help="The path to vocab file")

    args = parser.parse_args()

    vocab = read_vocab(args.vocab_path)
    vocab_size = len(vocab)

    model = Im2LatexModel(vocab_size, args.emb_dim, args.enc_rnn_h,
                          args.dec_rnn_h, beam_width=args.beam_width,
                          cnn_module=args.cnn_encoder, in_lstm_ch=args.in_lstm_ch,
                          disable_layer_3=args.disable_layer_3, disable_layer_4=args.disable_layer_4,
                          dis_last_conv=args.disable_last_conv)  # max_len

    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    encoder = model.get_encoder_wrapper()
    decoder = model.get_decoder_wrapper()
    dataset = Im2LatexDataset(args.data_path, args.dataset, 
                               
                              inp_channels=args.inp_channels)

    res_dir = os.path.dirname(args.res_encoder_name)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    toy_loader = DataLoader(
        dataset,
        collate_fn=partial(collate_fn, vocab.sign2id,  
            batch_transform=Compose([BatchTransformRescale(0.535, 0.535), BatchToTensor()]) ),
        #pin_memory=True if use_cuda else False,
        num_workers=0)
    for img_name, imgs, _, _ in toy_loader:
        # res = encoder(imgs)
        # print(len(res))
        if DEBUG_MO_1:
            torch.onnx.export(encoder, imgs, args.res_encoder_name,
                              opset_version=11,
                              input_names=['imgs'],
                              dynamic_axes={'imgs': {0: 'batch', 1: "channels", 2: "height", 3: "width"},
                                            'row_enc_out': {0: 'batch', 1: 'H', 2: 'W'},
                                            'hidden': {1: 'B', 2: "H"},
                                            'context': {1: 'B', 2: "H"},
                                            # 'init_0': {}
                                            },
                              output_names=['row_enc_out',
                                            'hidden',
                                            'context',
                                            # 'init_0'
                                            ],
                              verbose=False)

            exit(0)
        else:
            torch.onnx.export(encoder, imgs, args.res_encoder_name,
                              opset_version=11,
                              input_names=['imgs'],
                              dynamic_axes={'imgs': {0: 'batch', 1: "channels", 2: "height", 3: "width"},
                                            'row_enc_out': {0: 'batch', 1: 'H', 2: 'W'},
                                            'hidden': {1: 'B', 2: "H"},
                                            'context': {1: 'B', 2: "H"},
                                            'init_0': {}
                                            },
                              output_names=['row_enc_out', 'hidden', 'context', 'init_0'])
        if DEBUG_ONNX:
            row_enc_out_orig, h_orig, c_orig, init_0_orig = encoder(imgs)
            break
        logits_model, pred_model = model(imgs)
        print('image = {}'.format(img_name))
        break

    print("img shape: {}".format(imgs.shape))
    print("Shape of np.array imgs {}".format(np.array(imgs).shape))
    sess_pt1 = rt.InferenceSession(args.res_encoder_name)
    imgs_name = sess_pt1.get_inputs()[0].name
    row_enc_out_name = sess_pt1.get_outputs()[0].name
    h_name = sess_pt1.get_outputs()[1].name
    c_name = sess_pt1.get_outputs()[2].name
    init_0_name = sess_pt1.get_outputs()[3].name

    row_enc_out, h, c, O_t = sess_pt1.run([row_enc_out_name, h_name, c_name, init_0_name],
                                          {imgs_name: np.array(
                                              imgs).astype(np.float32)},
                                          )[:]  # [0:2]

    if DEBUG_ONNX:
        print("row_enc_out diff: {}".format(
            np.amax(abs(row_enc_out - row_enc_out_orig.detach().numpy()))))
        print("hidden diff:      {}".format(
            np.amax(abs(h - h_orig.detach().numpy()))))
        print("context dixff:    {}".format(
            np.amax(abs(c - c_orig.detach().numpy()))))
        print("init_0 diff:      {}".format(
            np.amax(abs(O_t - init_0_orig.detach().numpy()))))
        return 0

    print('row_enc_out shape = {}'.format(row_enc_out.shape))
    print('h shape = {}'.format(h.shape))
    print('c shape = {}'.format(c.shape))

    tgt = torch.tensor([[START_TOKEN]] * imgs.size(0))
    for _, imgs, tgt4train, tgt4loss in toy_loader:
        # print(O_t.shape)
        torch.onnx.export(decoder,
                          (torch.tensor(h),
                           torch.tensor(c),
                           torch.tensor(O_t),
                           torch.tensor(row_enc_out),
                           torch.tensor(tgt, dtype=torch.long)),
                          args.res_decoder_name,
                          opset_version=10,
                          input_names=['dec_st_h', 'dec_st_c',
                                       'output_prev', 'row_enc_out', 'tgt'],
                          output_names=['dec_st_h_t',
                                        'dec_st_c_t', 'output', 'logit'],
                          dynamic_axes={row_enc_out_name: {
                              0: 'batch', 1: 'H', 2: 'W'}}
                          )
        break
    # add_node_names("im2latex_model_decode_step.onnx")

    sess_pt3 = rt.InferenceSession(args.res_decoder_name)

    dec_st_h_name = sess_pt3.get_inputs()[0].name
    dec_st_c_name = sess_pt3.get_inputs()[1].name
    O_t_inp_name = sess_pt3.get_inputs()[2].name
    row_enc_out_name = sess_pt3.get_inputs()[3].name
    tgt_inp_name = sess_pt3.get_inputs()[4].name

    dec_st_out_h_name = sess_pt3.get_outputs()[0].name
    dec_st_out_c_name = sess_pt3.get_outputs()[1].name
    O_t_out_name = sess_pt3.get_outputs()[2].name
    logit_name = sess_pt3.get_outputs()[3].name

    # print(type(O_t))
    dec_states_out_h, dec_states_out_c, O_t_out_val, logit_val = sess_pt3.run(
        [
            dec_st_out_h_name,
            dec_st_out_c_name,
            O_t_out_name,
            logit_name
        ],
        {
            dec_st_h_name: h,
            dec_st_c_name: c,
            O_t_inp_name: O_t,
            row_enc_out_name: row_enc_out,
            tgt_inp_name: np.array(tgt)
        })[:]
    logits = []
    logits.append(logit_val)
    print('dec_states_out_h shape = {}'.format(dec_states_out_h.shape))
    print('dec_states_out_c shape = {}'.format(dec_states_out_c.shape))
    print('O_t_out_val shape = {}'.format(O_t_out_val.shape))
    print('logit_val shape = {}'.format(logit_val.shape))

    for t in range(args.max_formula_len):
        tgt = torch.reshape(torch.max(torch.tensor(logit_val), dim=1)[
                            1], (imgs.size(0), 1)).clone().detach()

        dec_states_out_h, dec_states_out_c, O_t_out_val, logit_val = sess_pt3.run(
            [
                dec_st_out_h_name,
                dec_st_out_c_name,
                O_t_out_name,
                logit_name
            ],
            {
                dec_st_h_name: dec_states_out_h,
                dec_st_c_name: dec_states_out_c,
                O_t_inp_name: O_t_out_val,
                row_enc_out_name: row_enc_out,
                tgt_inp_name: np.array(tgt)
            })[:]

        logits.append(logit_val)

    logits = torch.tensor(logits)
    logits = logits.squeeze(1)
    torch.set_printoptions(precision=2, profile='full', sci_mode=False)
    targets = torch.max(torch.log(torch.tensor(logits)).data, dim=1)[1]
    print("logits difference: ", torch.max(
        torch.abs(logits[0:100] - logits_model.squeeze(0)[0:100])))
    pred_phrase_str = vocab.construct_phrase(
        targets, args.max_formula_len)  # , args.max_formula_len)
    gold_phrase_str = vocab.construct_phrase(tgt4loss[0], args.max_formula_len)
    pred_phrase_model = vocab.construct_phrase(
        pred_model[0], args.max_formula_len)
    print("ONNX:   \t{}".format(pred_phrase_str))
    print("Gold:   \t{}".format(gold_phrase_str))
    print("Pytorch:\t{}".format(pred_phrase_model))


if __name__ == "__main__":
    main(sys.argv[1:])
