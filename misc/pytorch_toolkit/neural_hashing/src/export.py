from src.utils.exporter import Exporter
import argparse
from src.utils.get_config import get_config



def export(args):
    export_config = get_config(action = 'export')
    exporter = Exporter(export_config, openvino=1)

    if args.onnx:
        exporter.export_model_onnx(parent_dir = '')
    if args.ir:
        exporter.export_model_ir(parent_dir = '')

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

    custom_args = parser.parse_args()

    export(custom_args)
