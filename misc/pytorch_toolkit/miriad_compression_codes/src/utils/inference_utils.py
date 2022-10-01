import torch
import torchvision
import os
from torch import nn
import onnxruntime
from .model import AutoEncoder, Decoder
from .dataloader import CustomDatasetPhase1, CustomDatasetPhase2
from torch.utils import data
import numpy as np
import huffman
from .evaluators import compare_psnr_batch, compare_ssim_batch
import pickle
import argparse
from PIL import Image
from openvino.inference_engine import IECore
from torch.backends import cudnn
cudnn.benchmark = True

# The Float->Int module


class Float2Int(nn.Module):
    def __init__(self, bit_depth=8):
        super().__init__()
        self.bit_depth = bit_depth

    def forward(self, x):
        x = torch.round(x * (2**self.bit_depth - 1)).type(torch.int32)
        return x

# The Int->Float module


class Int2Float(nn.Module):
    def __init__(self, bit_depth=8):
        super().__init__()
        self.bit_depth = bit_depth

    def forward(self, x):
        x = x.type(torch.float32) / (2**self.bit_depth - 1)
        return x


def load_inference_model(config, run_type):
    if config['phase'] == 1:
        model = AutoEncoder(config['depth'], config['width'])
    else:
        model = Decoder(config['depth'], config['width'])

    # load the given model
    with open(os.path.abspath(config['model_file']), 'rb') as modelfile:
        loaded_model_file = torch.load(modelfile)
        model.load_state_dict(loaded_model_file['model_state'])
        # model.load_state_dict(loaded_model_file['model_state'])

    # return model
    if run_type == 'pytorch':
        pass
    elif run_type == 'onnx':
        model = onnxruntime.InferenceSession(config['checkpoint'])
    else:
        ie = IECore()
        model_xml = os.path.splitext(config['checkpoint'])[0] + ".xml"
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        model_temp = ie.read_network(model_xml, model_bin)
        model = ie.load_network(network=model_temp, device_name='CPU')

    model.eval()
    return model


def validate_model(model, config, run_type):

    if config['bit_depth'] == 16:
        float2int = Float2Int(12)
        int2float = Int2Float(12)
    else:
        float2int = Float2Int(config['bit_depth'])
        int2float = Int2Float(config['bit_depth'])

    # GPU transfer
    if torch.cuda.is_available() and config['gpu']:
        model = model.cuda()
        # model  = model.cuda()
        float2int, int2float = float2int.cuda(), int2float.cuda()

    images_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.Grayscale(),
         torchvision.transforms.ToTensor()])
    labels_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.Grayscale(),
         torchvision.transforms.ToTensor()])
    if config['phase'] == 1:
        infer_dataset = CustomDatasetPhase1(config['inferdata'],
                                            transform_images=images_transforms,
                                            transform_masks=labels_transforms,
                                            preserve_names=True)
        infer_dataloader = data.DataLoader(infer_dataset, batch_size=1,
                                           num_workers=16, pin_memory=True,
                                           shuffle=False)
    else:
        path_test_latent = config['path_to_latent']
        path_test_gdtruth = config['path_to_gdtruth']
        infer_dataset = CustomDatasetPhase2(path_to_latent=path_test_latent, path_to_gdtruth=path_test_gdtruth,
                                            transform_images=images_transforms, transform_masks=labels_transforms, preserve_name=True, mod=0)
        infer_dataloader = data.DataLoader(
            infer_dataset, batch_size=1, num_workers=16, pin_memory=True, shuffle=False)

    with torch.no_grad():
        # a global counter & accumulation variables
        n = 0
        all_bits, all_bpp, all_ssim, all_psnr = [], [], [], []

        for idx, (image, name) in enumerate(infer_dataloader):
            if torch.cuda.is_available() and config['gpu']:
                image = image.cuda()
            dtype1 = image.dtype
            compressed = model.encoder(image)  # forward through encoder
            # forward through Float2Int module
            latent_int = float2int(compressed)

            # usual numpy conversions
            image_numpy = image.cpu().numpy()
            latent_int_numpy = latent_int.cpu().numpy()

            if config['with_aac']:
                # encode latent_int with Huffman coding

                # calculate the source symbol distribution; required for huffman encoding
                if config['bit_depth'] == 16:
                    counts, bins = np.histogram(
                        latent_int_numpy.ravel(), bins=2**12 - 1)
                else:
                    ap = argparse.ArgumentParser()
                    args = ap.parse_args()
                    counts, bins = np.histogram(
                        latent_int_numpy.ravel(), bins=2**args.bit_depth - 1)
                z = list(iter(zip(bins.tolist(), counts.tolist())))
                Q = huffman.codebook(z)  # Huffman codebookth
                H = 0  # calculate the entropy of the codebook
                for i, (_, bitstr) in enumerate(Q.items()):
                    H += counts[i]/counts.sum() * len(bitstr)

                # this is actual bpp to be reported
                bits = H
            else:
                # if not --with_aac, there is no codebook
                # in that case, bpp is the bit depth of latent integer tensor
                bits = config['bit_depth']
                Q = None

            # the canonical formula for calculating BPP
            bpp = latent_int_numpy.size * bits / image_numpy.size

            latent_float = int2float(latent_int)  # back to float
            decompressed = model.decoder(
                latent_float)  # forward through decoder
            original, reconstructed = image_numpy, decompressed.cpu().numpy()

            # computer required metrics
            ssim = compare_ssim_batch(original, reconstructed)
            psnr = compare_psnr_batch(original, reconstructed)
            psnr = 20.0 * np.log10(psnr)

            # averaging
            all_bits.append(bits)
            all_bpp.append(bpp)
            all_ssim.append(ssim)
            all_psnr.append(psnr)

            n += 1

            if config['produce_latent_code']:
                # save the latent code if requested. the saved items are
                # 1. The entire latent integer tensor
                # 2. The codebook after running AAC
                latent_file_path = os.path.join(
                    os.path.abspath(config['out_latent']),
                    'cbis_' + str(config['bit_depth']) + '_' + name[0] + '.latent')
                with open(latent_file_path, 'wb') as latent_file:
                    pickle.dump({
                        'latent_int': latent_int.cpu().numpy(),
                        'bits': bits,
                        'codebook': Q
                    }, latent_file)

            if config['produce_decompressed_image']:
                # save the reconstructed image, if requested
                reconstructed = reconstructed.squeeze()
                reconstructed = Image.fromarray(
                    reconstructed * 255.).convert('L')
                decom_file = os.path.join(
                    os.path.abspath(config['out_decom']),
                    str(config['bit_depth']) + '_decom_' + name[0])
                reconstructed.save(decom_file)

            print('name: {}, bit_depth: {}, bpp: {}, ssim: {}, psnr: {}, dtype: {}'.format(name[0],
                                                                                           config['bit_depth'], bpp, ssim, psnr, dtype1))

            if n == config['max_samples']:
                break

            # write the metrics for plotting
            import json
            json_content = []
            json_fillpath = os.path.abspath(config['plot_json'])
            if not os.path.exists(json_fillpath):
                with open(json_fillpath, 'w') as json_file:
                    json.dump([], json_file)

            with open(json_fillpath, 'r') as json_file:
                json_content = json.load(json_file)

            # append to the content of json
            json_content.append(
                {'bits': all_bits, 'bpp': all_bpp, 'ssim': all_ssim, 'psnr': all_psnr})

            with open(json_fillpath, 'w') as json_file:
                json.dump(json_content, json_file)

    return np.mean(all_ssim), np.mean(all_psnr)
