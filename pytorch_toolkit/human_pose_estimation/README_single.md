# Global Context for Convolutional Pose Machines

This repository contains training code for the paper [Global Context for Convolutional Pose Machines](https://arxiv.org/pdf/1906.04104.pdf). This work improves [original](https://arxiv.org/pdf/1602.00134.pdf) convolutional pose machine architecture for artculated human pose estimation in both accuracy and inference speed. On the Look Into Person (LIP) test set this code achives 87.9% PCKh for a single model, the 2-stage version of this network runs with more than 160 frames per second on a GPU and ~20 frames per second on a CPU. The result can be reproduced using this repository.

## Prerequisites

1. Download the [Look Into Person dataset](http://47.100.21.47:9999/overview.php) and unpack it to `<LIP_HOME>` folder.
2. Create virtual environment `bash init_venv.sh`

## Training

1. Download pre-trained MobileNet v1 weights `mobilenet_sgd_68.848.pth.tar` from: [https://github.com/marvis/pytorch-mobilenet](https://github.com/marvis/pytorch-mobilenet) (sgd option).
2. Run in terminal: `python train_single.py --dataset-folder <LIP_HOME> --checkpoint-path mobilenet_sgd_68.848.pth.tar --from-mobilenet`

## Validation
1. Run in terminal `python val_single.py --dataset-folder <LIP_HOME> --checkpoint-path <CHECKPOINT>`. One should observe ~84% PCKh on validation set (use `--multiscale` and set `flip` to `True` for better results).

  [OPTIONAL] Pass `--visualize` key to see predicted keypoints results.

The final number on the test set was obtained with addition of validation data into training.

## Conversion to OpenVINO format:

1. Convert PyTorch model to ONNX format: run script in terminal `python scripts/convert_to_onnx.py --checkpoint-path <CHECKPOINT> --single-person`. It produces `human-pose-estimation.onnx`.
2. Convert ONNX model to OpenVINO format with Model Optimizer: run in terminal `python <OpenVINO_INSTALL_DIR>/deployment_tools/model_optimizer/mo.py --input_model human-pose-estimation.onnx --input data --mean_values data[128.0,128.0,128.0] --scale_values data[256] --output stage_1_output_1_heatmaps`. This produces model `human-pose-estimation.xml` and weights `human-pose-estimation.bin` in single-precision floating-point format (FP32).

## Citation:

If this helps your research, please cite the papers:

```
@inproceedings{osokin2019global_context_cpm,
    author={Osokin, Daniil},
    title={Global Context for Convolutional Pose Machines},
    booktitle = {arXiv preprint arXiv:1906.04104},
    year = {2019}
}

@inproceedings{osokin2018lightweight_openpose,
    author={Osokin, Daniil},
    title={Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose},
    booktitle = {arXiv preprint arXiv:1811.12004},
    year = {2018}
}
```

