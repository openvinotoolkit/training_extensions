from utils.exporter import Exporter
import argparse
from utils.get_config import get_config

def export(args):
    export_config = get_config(action='export', phase=args.phase)
    exporter = Exporter(export_config, args.optimised)

    if args.onnx:
        exporter.export_model_onnx()
    if args.ir:
        exporter.export_model_ir()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx",
                        required=False,
                        help="Set to True, if you wish to export onnx model",
                        default=False,
                        action='store_true')
    parser.add_argument("--ir",
                        required=False,
                        help="Set to True, if you wish to export IR",
                        default=False,
                        action='store_true')
    parser.add_argument('-ph', '--phase', type=int,
                        required=True, default=1, help='Phase')

    custom_args = parser.parse_args()

    export(custom_args)
