from ..utils.exporter import Exporter
import argparse
from src.utils.utils import load_json


def export(args):
    config = load_json(namespace.config_filename)
    config["model_name"] = namespace.model_name
    config["model_onnx_filename"] = namespace.model_filename.split('.')[0]+".onnx"
    exporter = Exporter(config)

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
    parser.add_argument("--model_filename", 
        required=True)
    parser.add_argument("--config_filename", 
        required=True)
    parser.add_argument("--model_name",
        help="Specifies the model you want to use",
        default="", type=str)

    namespace = parser.parse_args()

    export(namespace)
