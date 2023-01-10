import argparse
from .utils.exporter import Exporter
from .utils.get_config import get_config

def export(configs):
    export_config = get_config(action='export', stage=configs["stage"])
    exporter = Exporter(export_config, stage=configs["stage"])

    if configs["onnx"]:
        exporter.export_model_onnx()
    if configs["ir"]:
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
    parser.add_argument('-s', '--stage', type=int,
                        required=True, default=1, help='Stage')

    args = parser.parse_args()

    configs = {
        "onnx": args.onnx,
        "ir": args.ir,
        "stage": args.stage

   }
    export(configs)
