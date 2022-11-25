import torch
import torchvision
import os
from torch import nn
import onnxruntime
from .model import AutoEncoder, Decoder
from .dataloader import CustomDatasetPhase1, CustomDatasetPhase2
from torch.utils import data
import numpy as np
from dahuffman import HuffmanCodec
from .evaluators import compare_psnr_batch, compare_ssim_batch
import pickle
import sys
import argparse
from PIL import Image
from openvino.inference_engine import IECore
from torch.backends import cudnn
import json
import time
cudnn.benchmark = True

class Float2Int(nn.Module):
	def __init__(self, bit_depth=8):
		super().__init__()
		self.bit_depth = bit_depth

	def forward(self, x):
		x = torch.round(x * (2**self.bit_depth - 1)).type(torch.int32)
		return x

class Int2Float(nn.Module):
	def __init__(self, bit_depth=8):
		super().__init__()
		self.bit_depth = bit_depth

	def forward(self, x):
		x = x.type(torch.float32) / (2**self.bit_depth - 1)
		return x

def load_inference_model(config, run_type):

	if run_type == 'pytorch':
		if config['phase'] == 1:
			model = AutoEncoder(config['depth'], config['width'])
		else:
			model = Decoder(config['depth'], config['width'])

		with open(os.path.abspath(config['model_file']), 'rb') as modelfile:
			loaded_model_file = torch.load(modelfile,map_location=torch.device('cpu'))
			model.load_state_dict(loaded_model_file['model_state'])
		model.eval()

	elif run_type == 'onnx':
		model = onnxruntime.InferenceSession(os.path.splitext(config['checkpoint'])[0] + ".onnx")

	else:
		ie = IECore()
		split_text = os.path.splitext(config['checkpoint'])[0]
		model_xml =  split_text + ".xml"
		model_bin = split_text + ".bin"
		model_temp = ie.read_network(model_xml, model_bin)
		model = ie.load_network(network=model_temp, device_name='CPU')

	return model

def validate_model(model, config, run_type):

	if config['bit_depth'] == 16:
		float2int = Float2Int(12)
		int2float = Int2Float(12)
	else:
		float2int = Float2Int(config['bit_depth'])
		int2float = Int2Float(config['bit_depth'])

	# GPU transfer - Only pytorch models needs to be transfered.
	if run_type == 'pytorch':
		if torch.cuda.is_available() and config['gpu']:
			model = model.cuda()
			float2int, int2float = float2int.cuda(), int2float.cuda()
	else:
		pass

	custom_transforms = torchvision.transforms.Compose(
		[torchvision.transforms.Grayscale(),
		 torchvision.transforms.ToTensor()])

	if config['phase'] == 1:
		infer_dataset = CustomDatasetPhase1(config['inferdata'],
											transform_images=custom_transforms,
											transform_masks=custom_transforms,
											preserve_names=True)
		infer_dataloader = data.DataLoader(infer_dataset, batch_size=1,
										   num_workers=16, pin_memory=True,
										   shuffle=False)
	else:
		path_test_latent = config['path_to_latent']
		path_test_gdtruth = config['path_to_gdtruth']
		infer_dataset = CustomDatasetPhase2(path_to_latent=path_test_latent, path_to_gdtruth=path_test_gdtruth,
											transform_images=custom_transforms, transform_masks=custom_transforms,
											preserve_name=True, mod=0)
		infer_dataloader = data.DataLoader(
			infer_dataset, batch_size=1, num_workers=16, pin_memory=True, shuffle=False)

	with torch.no_grad():
		# a global counter & accumulation variables
		n = 0
		all_bits, all_bpp, all_ssim, all_psnr = [], [], [], []
		all_cf = []
		all_bits = []
		# all_bits, all_cf, all_ssim, all_psnr = [], [], [], []
		avg_cf, avg_ssim, avg_psnr,avg_bits = 0., 0., 0., 0.

		all_netc_tim, all_enc_huf_tim, all_dec_huf_tim,all_netd_tim = [], [], [], []
		avg_netc_tim, avg_enc_huf_tim,avg_dec_huf_tim,avg_netd_tim = 0., 0., 0., 0.

		for idx, (image, name) in enumerate(infer_dataloader):
			img_size = sys.getsizeof(image.storage())
			if torch.cuda.is_available() and config['gpu']:
				image = image.cuda()
			dtype1 = image.dtype
			if run_type == 'pytorch':
				compressed = model.encoder(image)  # forward through encoder
			else:
				compressed = model(image)



			# forward through Float2Int module
			latent_int = float2int(compressed)
			# usual numpy conversions
			image_numpy = image.cpu().numpy()
			latent_int_numpy = latent_int.cpu().numpy()

			if config["with_aac"]:
				# encode latent_int with Huffman coding

				# calculate the source symbol distribution; required for huffman encoding
				inpt=[]
				# print(latent_int_numpy.shape)
				c1, c, h, w = latent_int_numpy.shape
				flat = latent_int_numpy.flatten()
				for i in flat:
					inpt.append(str(i))

				codec = HuffmanCodec.from_data(inpt)
				encoded = codec.encode(inpt)

				hufbook = codec.get_code_table()
				book_size = sys.getsizeof(hufbook)
				code_size = sys.getsizeof(encoded)
				# print(book_size)
				cf = img_size/(book_size+code_size)
				# print("compression factor:",cf)

				all_cf.append(cf)
				bits_al = []
				for symbol, (bits, val) in hufbook.items():
					bits_al.append(bits)
				bits_al = np.array(bits_al)
				av_bits = np.mean(bits_al)
				all_bits.append(av_bits)
				decoded = codec.decode(encoded)

				# print("AAC-dec :",end-start)
				# all_dec_huf_tim.append(end-start)
				ar_in=[]
				for i in decoded:
					ar_in.append(int(i))
				ar_in = np.array(ar_in)
				latent = ar_in.reshape([c1,c,h,w])
				latent_inp = torch.from_numpy(latent).cuda()
			else:
				# if not --with_aac, there is no codebook
				# in that case, bpp is the bit depth of latent integer tensor
				bits = config["bit_depth"]
				Q = None

			# the canonical formula for calculating BPP
			bpp = latent_int_numpy.size * bits / image_numpy.size

			latent_float = int2float(latent_inp)  # back to float
			decompressed = model.decoder(latent_float)  # forward through decoder
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
