import argparse
import os
import torch.onnx
import train

parser = argparse.ArgumentParser(description="PyTorch SR export to onnx")
parser.add_argument("--test_data_path", default="", type=str, help="path to test data")
parser.add_argument("--exp_name", default="test", type=str, help="experiment name")
parser.add_argument("--models_path", default="/models", type=str, help="path to models folder")
parser.add_argument("--input_size", type=int, nargs='+', default=(200, 200), help="Input image size")
parser.add_argument("--scale", type=int, default=4, help="Upsampling factor for SR")


def main():
    opt = parser.parse_args()

    models_path = opt.models_path
    name = opt.exp_name
    input_size = opt.input_size
    scale = opt.scale

    x = torch.randn(1, 3, input_size[0], input_size[1], requires_grad=True).cuda()
    cubic = torch.randn(1, 3, scale*input_size[0], scale*input_size[1], requires_grad=True).cuda()

    trainer = train.Trainer(name=name, models_root=models_path, resume=True)
    trainer.load_best()

    trainer.model = trainer.model.train(False)

    torch_out = torch.onnx.export(trainer.model,  # model being run
                                  [x, cubic],  # model input (or a tuple for multiple inputs)
                                  os.path.join(models_path, name, "model.onnx"),  # where to save the model (can be a file or file-like object)
                                  export_params=True,
                                  verbose=True)  # store the trained parameter weights inside the model file

if __name__ == "__main__":
    main()


