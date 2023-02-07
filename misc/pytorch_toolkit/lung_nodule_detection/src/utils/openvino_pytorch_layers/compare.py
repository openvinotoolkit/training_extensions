# NOTE: import order is critical for now: extensions, openvino and only then numpy
from openvino_extensions import get_extensions_path
from openvino.inference_engine import IECore

import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Compare OpenVINO implementation with reference data')
parser.add_argument('--num_inputs', type=int, default=1)
parser.add_argument('-m', '--model', default="model.xml")
parser.add_argument('-d', '--device', default="CPU")
args = parser.parse_args()

inputs = {}
shapes = {}
for i in range(args.num_inputs):
    suffix = '{}'.format(i if i > 0 else '')
    data = np.load('inp' + suffix + '.npy')
    inputs['input' + suffix] = data
    shapes['input' + suffix] = data.shape

ref = np.load('ref.npy')

ie = IECore()
ie.add_extension(get_extensions_path(), 'CPU')
ie.set_config({'CONFIG_FILE': 'user_ie_extensions/gpu_extensions.xml'}, 'GPU')

net = ie.read_network(args.model)
net.reshape(shapes)
exec_net = ie.load_network(net, args.device)

out = exec_net.infer(inputs)
out = next(iter(out.values()))

maxdiff = np.max(np.abs(ref - out))
print('Reference range: [{}, {}]'.format(np.min(ref), np.max(ref)))
print('Maximal difference:', maxdiff)
if maxdiff > 1e-5:
    exit(1)
