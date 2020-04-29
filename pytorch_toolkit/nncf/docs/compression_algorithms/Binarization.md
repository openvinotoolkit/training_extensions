### Binarization
NNCF supports binarizing weights and activations for 2D convolutional PyTorch\* layers (Conv2D) *only*.

Weight binarization may be done in two ways, depending on the configuration file parameters - either via [XNOR binarization](https://arxiv.org/abs/1603.05279) or via [DoReFa binarization](https://arxiv.org/abs/1606.06160). For DoReFa binarization, the scale of binarized weights for each convolution operation is calculated as the mean of absolute values of non-binarized convolutional filter weights, while for XNOR binarization, each convolutional operation has scales that are calculated in the same manner, but _per input channel_ of the convolutional filter. Refer to the original papers for details.

Binarization of activations is implemented via binarizing inputs to the convolutional layers in the following way:

![\text{out} = s * H(\text{in} - s*t)](https://microsoft.codecogs.com/png.latex?%5Ctext%7Bout%7D%20%3D%20s%20*%20H%28%5Ctext%7Bin%7D%20-%20s*t%29)

In the formula above,
 - ![\text{in}](https://microsoft.codecogs.com/png.latex?%5Ctext%7Bin%7D) - non-binarized activation values
 - ![\text{out}](https://microsoft.codecogs.com/png.latex?%5Ctext%7Bout%7D) - binarized activation values
 -  ![H(x)](https://microsoft.codecogs.com/png.latex?H%28x%29) is the Heaviside step function
 - ![s](https://microsoft.codecogs.com/png.latex?s) and ![t](https://microsoft.codecogs.com/png.latex?t) are trainable parameters corresponding to binarization scale and threshold respectively

Training binarized networks requires special scheduling of the training process. For instance, binarizing a pretrained ResNet18 model on ImageNet is a four-stage process, with each stage taking a certain number of epochs. During the stage 1, the network is trained without any binarization. During the stage 2, the training continues with binarization enabled for activations only. During the stage 3, binarization is enabled both for activations and weights. Finally, during the stage 4 the optimizer learning rate, which was kept constant at previous stages, is decreased according to a polynomial law, while weight decay parameter of the optimizer is set to 0. The configuration files for the NNCF binarization algorithm allow to control certain parameters of this training schedule.


**Algorithm configuration file parameters**:

```
{
    "algorithm": "binarization",
    "mode": "xnor", // Selects the mode of binarization - either 'xnor' for XNOR binarization, or 'dorefa' for DoReFa binarization
    "params": {
        "batch_multiplier": 1,  // Gradients will be accumulated for this number of batches before doing a 'backward' call
        "activations_bin_start_epoch": 10,  // Epoch to start binarizing activations
        "weights_bin_start_epoch": 30,  // Epoch to start binarizing weights
        "lr_poly_drop_start_epoch": 60,  // Epoch to start dropping the learning rate
        "lr_poly_drop_duration_epochs": 30,  // Duration, in epochs, of the learning rate dropping process.
        "disable_wd_start_epoch": 60  // Epoch to disable weight decay in the optimizer
    },

    //A list of model control flow graph node scopes to be ignored for this operation - functions as a 'blacklist'. Optional.
    "ignored_scopes": ["ResNet/Linear[fc]",
                       "ResNet/Conv2d[conv1]",
                       "ResNet/Sequential[layer2]/BasicBlock[0]/Sequential[downsample]",
                       "ResNet/Sequential[layer3]/BasicBlock[0]/Sequential[downsample]",
                       "ResNet/Sequential[layer4]/BasicBlock[0]/Sequential[downsample]"],

    // A list of model control flow graph node scopes to be considered for this operation - functions as a 'whitelist'. Optional.
    // "target_scopes": [],
}

```
