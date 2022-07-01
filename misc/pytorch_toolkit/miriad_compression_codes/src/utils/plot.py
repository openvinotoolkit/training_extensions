import matplotlib.pyplot as plt
import json

def main( args ):
	with open(args.plot_json, 'r') as json_file:
		# load the .json file
		plot_points = json.load(json_file)

	# extract the 'bpp', 'ssim' & 'psnr' fields
	bpp  = [point['bpp'] for point in plot_points]
	ssim = [point['ssim'] for point in plot_points]
	psnr = [point['psnr'] for point in plot_points]

	# plot them with usual plt.xxplot() calls
	plt.figure()
	plt.subplot(1, 2, 1); plt.scatter(bpp, ssim); plt.xlabel('bpp'); plt.ylabel('MS-SSIM')
	plt.title('MS-SSIM vs bpp (bits/pixel)')
	plt.subplot(1, 2, 2); plt.scatter(bpp, psnr); plt.xlabel('bpp'); plt.ylabel('pSNR (dB)')
	plt.title('pSNR vs bpp (bits/pixel)')

	plt.savefig(args.plot_image)

if __name__ == '__main__':
	import argparse
	parse = argparse.ArgumentParser(
	"""
	This plotting script generates the SSIM-vs-bpp and pSNR-vs-bpp plots.

	Provide the json file produced by inference.py.
	(Optionally) Provide name of the plot image. Default is plot.png.
	"""
	)

	parse.add_argument('-p', '--plot_json', type=str, required=True, help='the plot json file produced by inference script')
	parse.add_argument('-o', '--plot_image', type=str, required=False, default='plot.png', help='name of plot image file')
	args = parse.parse_args()

	main( args )
