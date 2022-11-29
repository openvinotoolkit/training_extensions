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
from PIL import Image
from openvino.inference_engine import IECore
from torch.backends import cudnn
import json
cudnn.benchmark = True
from torchvision import transforms

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

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def lossless_compression(latent_int_numpy, config,img_size):
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
	bits_al = []
	for symbol, (bits, val) in hufbook.items():
		bits_al.append(bits)
		bits_al_np = np.array(bits_al)
		av_bits = np.mean(bits_al_np)
		decoded = codec.decode(encoded)

		ar_in=[]
		for i in decoded:
			ar_in.append(int(i))
		ar_in = np.array(ar_in)
		latent = ar_in.reshape([c1,c,h,w])
		latent_inp = torch.from_numpy(latent) #.cuda()
	
	return bits, latent_inp, hufbook


def load_inference_model(config, run_type):

	if run_type == 'pytorch':
		if config['phase'] == 1:
			model = AutoEncoder(config['depth'], config['width'])
		else:
			model = Decoder(config['depth'], config['width'])

		with open(os.path.abspath(config['model_file']), 'rb') as modelfile:
			loaded_model_file = torch.load(modelfile, map_location=torch.device('cpu'))
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

	if not os.path.exists('temp_data'):
		os.makedirs('temp_data')
		os.makedirs(os.path.join('temp_data','phase1','latent'))
		os.makedirs(os.path.join('temp_data','phase1','decom'))
		os.makedirs(os.path.join('temp_data','phase2','latent'))
		os.makedirs(os.path.join('temp_data','phase2','decom'))

	if config['bit_depth'] == 16:
		float2int = Float2Int(12)
		int2float = Int2Float(12)
	else:
		float2int = Float2Int(config['bit_depth'])
		int2float = Int2Float(config['bit_depth'])

	# GPU transfer - Only pytorch models needs to be transfered.
	if run_type == 'pytorch':
		if torch.cuda.is_available() and config['gpu'] == 'True':
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
		infer_dataset = CustomDatasetPhase2(path_to_latent=config['path_to_latent'], path_to_gdtruth=config['path_to_gdtruth'],
											transform_images=custom_transforms, transform_masks=custom_transforms,
											preserve_name=False, mod=0)
		infer_dataloader = data.DataLoader(infer_dataset, batch_size=1, num_workers=16, pin_memory=True, shuffle=False)

	with torch.no_grad():
		# a global counter & accumulation variables
		n = 0
		all_ssim, all_psnr = [], []


		for data_list in infer_dataloader:

			tensor_1 = data_list[0]
			print(tensor_1.shape)
			tensor_2 = data_list[1]
			file_name = data_list[2]
			img_size = sys.getsizeof(tensor_1.storage())
			if torch.cuda.is_available() and config['gpu'] == 'True':
				tensor_1 = tensor_1.cuda()

			if run_type == 'pytorch':
				if config['phase']==1:
					compressed = model.encoder(tensor_1)  # forward through encoder

					latent_int = float2int(compressed)
					# usual numpy conversions
					tensor_1_numpy = tensor_1.cpu().numpy()
					latent_int_numpy = latent_int.cpu().numpy()

					if config["with_aac"] == 'True':

						bits, latent_inp, hufbook = lossless_compression(latent_int_numpy,config,img_size)
						# the canonical formula for calculating BPP
						bpp = latent_int_numpy.size * bits / tensor_1_numpy.size

						latent_float = int2float(latent_inp)  # back to float
						decompressed = model.decoder(latent_float)  # forward through decoder
						original, reconstructed = tensor_1, decompressed
					else:
						# if not --with_aac, there is no codebook
						# in that case, bpp is the bit depth of latent integer tensor
						bits = config["bit_depth"]
						hufbook = None
				else:
					reconstructed = model(tensor_1)
					original = tensor_2

			elif run_type == 'onnx':
				original = tensor_1 if config['phase'] == 1 else tensor_2
				if config['phase'] == 1:
					resize_tensor = transforms.Resize([1024,1024]) # Size is set since the onnx model accepts fixed size
					tensor_1_ort = resize_tensor(tensor_1)
					original = resize_tensor(original)
				else:
					tensor_1_ort = tensor_1
					# print(tensor_1_ort.shape)
				ort_inputs = {model.get_inputs()[0].name: to_numpy(tensor_1_ort)}
				reconstructed = model.run(None, ort_inputs)
				to_tensor = transforms.ToTensor()
				reconstructed = np.array(reconstructed)
				reconstructed = np.squeeze(reconstructed,axis=0)
				reconstructed = to_tensor(np.squeeze(reconstructed,axis=0)).transpose(dim0=1, dim1=0).unsqueeze(0)
				bits, latent_int, hufbook = None, None, None

			else:
				original = tensor_1 if config['phase'] == 1 else tensor_2
				if config['phase'] == 1:
					resize_tensor = transforms.Resize([1024,1024]) # Size is set since the onnx model accepts fixed size
					tensor_1 = resize_tensor(tensor_1)
					original = resize_tensor(original)
				else:
					tensor_1_ort = tensor_1
				# resize_tensor = transforms.Resize([1024,1024])
				to_tensor = transforms.ToTensor()
				reconstructed = model.infer(inputs={'input': resize_tensor(tensor_1)})['output']
				reconstructed = np.array(reconstructed)
				reconstructed = np.squeeze(reconstructed,axis=0)
				reconstructed = to_tensor(np.squeeze(reconstructed,axis=0)).unsqueeze(0)

				bits, latent_int, hufbook = None, None, None

			ssim = compare_ssim_batch(original, reconstructed)
			psnr = compare_psnr_batch(original, reconstructed)

			# averaging
			all_ssim.append(ssim)
			all_psnr.append(psnr)

			n += 1

			if config['produce_latent_code'] == 'True':
				# save the latent code if requested. the saved items are
				# 1. The entire latent integer tensor
				# 2. The codebook after running AAC
				latent_file_path = os.path.join(
					os.path.abspath(config['out_latent']),
					'cbis_' + str(config['bit_depth']) + '_' + file_name[0] + '.latent')
				with open(latent_file_path, 'wb') as latent_file:
					pickle.dump({
						'latent_int': latent_int,
						'bits': bits,
						'codebook': hufbook
					}, latent_file)

			if config['produce_decompressed_image'] == 'True':
				# save the reconstructed , if requested
				reconstructed = reconstructed.squeeze().numpy()
				reconstructed = Image.fromarray(reconstructed * 255.).convert('L')
				decom_file = os.path.join(
					os.path.abspath(config['out_decom']),
					str(config['bit_depth']) + '_decom_' + file_name[0])
				reconstructed.save(decom_file)
			
			print(f'SSIM: {ssim}, PSNR: {psnr}')

			if n == config['max_samples']:
				break

	return np.mean(all_ssim), np.mean(all_psnr)
