# Neural Network Compression Framework  (NNCF)

This is a PyTorch\*-based framework for neural networks compression.

## Key Features

- Support of quantization, binarization, and sparsity algorithms with fine-tuning
- Automatic model graph transformation. The model is wrapped by the custom class and additional layers are inserted in the graph. The transformations are configurable.
- Common interface for compression methods
- GPU-accelerated layers for fast model fine-tuning
- Distributed training support
- Configuration file examples for sparsity, quantization, and sparsity with quantization. Each type of compression requires only one additional fine-tuning stage.
- Export models to ONNX\* format that is supported by the [OpenVINO&trade; toolkit](https://github.com/opencv/dldt).

## Compression Algorithm Architecture Overview

The compression method is divided into three logical parts:
- The algorithm itself that implements API of `CompressionAlgorithm` class and controls compression algorithm logic.
- The scheduler that implements API of `CompressionScheduler` class and compression method scheduling control logic.
- The loss that implements API of `CompressionLoss` class and represents auxiliary loss function which is required by the compression method. It must be added to the main objective function responsible for accuracy metric and trained jointly.

> **NOTE**: In general, the compression method may not have its own scheduler and loss, and the default implementations are used instead.

For more details, see the [Algorithm API](../docs/Algorithm.md) description manual.

## Implemented Compression Methods

Each compression method receives its own hyperparameters that are organized as a dictionary and basically stored in a JSON file that is deserialized when the training starts. Compression methods can be applied separately or together producing sparse, quantized, or both sparse and quantized models. For more information about the configuration, refer to the samples.

### Uniform Quantization with Fine-Tuning

A uniform "fake" quantization method supports an arbitrary number of bits (>=2) which is used to represent weights and activations.
The method performs differentiable sampling of the continuous signal (for example, activations or weights) during forward pass, simulating inference with integer arithmetic.

#### Common Quantization Formula

Quantization is parametrized by clamping range and number of quantization levels. The sampling formula is the following:

![output = \frac{\left\lfloor (clamp(input; input\_low, input\_high)-input\_low)  *s\right \rceil}{s} + input\_low\\](https://latex.codecogs.com/gif.latex?output%20%3D%20%5Cfrac%7B%5Cleft%5Clfloor%20%28clamp%28input%3B%20input%5C_low%2C%20input%5C_high%29-input%5C_low%29%20*s%5Cright%20%5Crceil%7D%7Bs%7D%20&plus;%20input%5C_low%5C%5C) 

![clamp(input; input\_low, input\_high) = min(max(input, input\_low), input\_high)))](https://latex.codecogs.com/gif.latex?clamp%28input%3B%20input%5C_low%2C%20input%5C_high%29%20%3D%20min%28max%28input%2C%20input%5C_low%29%2C%20input%5C_high%29%29%29)

![s=\frac{levels-1}{input\_high - input\_low}](https://latex.codecogs.com/gif.latex?s%3D%5Cfrac%7Blevels-1%7D%7Binput%5C_high%20-%20input%5C_low%7D) 

`input_low` and `input_high` represent the quantization range and ![\left\lfloor\cdot\right \rceil](https://latex.codecogs.com/gif.latex?%5Cleft%5Clfloor%5Ccdot%5Cright%20%5Crceil) denotes rounding to the nearest integer.


####  Symmetric Quantization

During the training, we optimize the **scale** parameter that represents the range `[input_low, input_range]` of the original signal using gradient descent:

![input\_low=scale*\frac{level\_low}{level\_high}](https://latex.codecogs.com/gif.latex?input%5C_low%3Dscale*%5Cfrac%7Blevel%5C_low%7D%7Blevel%5C_high%7D)

![input\_high=scale](https://latex.codecogs.com/gif.latex?input%5C_high%3Dscale)

In the formula above, `level_low` and `level_high` represent the range of the discrete signal.
 - For weights: 
 
    ![level\_low=-2^{bits-1}+1](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cdpi%7B120%7D%20level%5C_low%3D-2%5E%7Bbits-1%7D&plus;1), 
    
    ![level\_high=2^{bits-1}-1](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cdpi%7B120%7D%20level%5C_high%3D2%5E%7Bbits-1%7D-1)

    ![levels=255](https://latex.codecogs.com/gif.latex?levels%3D255)
 
 - For unsigned activations: 
 
    ![level\_low=0](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cdpi%7B120%7D%20level%5C_low%3D0)
    
    ![level\_high=2^{bits}-1](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cdpi%7B120%7D%20level%5C_high%3D2%5E%7Bbits%7D-1)
    
    ![levels=256](https://latex.codecogs.com/gif.latex?levels%3D256)

 - For signed activations: 
 
    ![level\_low=-2^{bits-1}](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20level%5C_low%3D-2%5E%7Bbits-1%7D)
    
    ![level\_high=2^{bits-1}-1](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cdpi%7B120%7D%20level%5C_high%3D2%5E%7Bbits-1%7D-1)
    
    ![levels=256](https://latex.codecogs.com/gif.latex?levels%3D256)

For all the cases listed above, the common quantization formula is simplified after substitution of `input_low`, `input_high` and `levels`:

![output = \left\lfloor clamp(input * \frac{level\_high}{scale}, level\_low, level\_high)\right \rceil * \frac{scale}{level\_high}](https://latex.codecogs.com/gif.latex?output%20%3D%20%5Cleft%5Clfloor%20clamp%28input%20*%20%5Cfrac%7Blevel%5C_high%7D%7Bscale%7D%2C%20level%5C_low%2C%20level%5C_high%29%5Cright%20%5Crceil%20*%20%5Cfrac%7Bscale%7D%7Blevel%5C_high%7D)

Use the `num_init_steps` parameter from the `initializer` group to initialize the values of `scale` and determine which activation should be signed or unsigned from the collected statistics during given number of steps.

####  Asymmetric Quantization

During the training we optimize the **input_low** and **input_range** parameters using gradient descent:

![input\_high=input\_low + input\_range](https://latex.codecogs.com/gif.latex?input%5C_high%3Dinput%5C_low%20&plus;%20input%5C_range)
 
![levels=256](https://latex.codecogs.com/gif.latex?levels%3D256)

![level\_low=0](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cdpi%7B120%7D%20level%5C_low%3D0)
 
![level\_high=2^{bits}-1](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cdpi%7B120%7D%20level%5C_high%3D2%5E%7Bbits%7D-1)

For better accuracy, floating-point zero should be within quantization range and strictly mapped into quant (without rounding). Therefore, the following scheme is applied to ranges of weights and activations before quantization:

![{input\_low}' = min(input\_low, 0)](https://latex.codecogs.com/gif.latex?%7Binput%5C_low%7D%27%20%3D%20min%28input%5C_low%2C%200%29)

![{input\_high}' = max(input\_high, 0)](https://latex.codecogs.com/gif.latex?%7Binput%5C_high%7D%27%20%3D%20max%28input%5C_high%2C%200%29)

![ZP= \left\lfloor \frac{-{input\_low}'*(levels-1)}{{input\_high}'-{input\_low}'} \right \rceil ](https://latex.codecogs.com/gif.latex?ZP%3D%20%5Cleft%5Clfloor%20%5Cfrac%7B-%7Binput%5C_low%7D%27*%28levels-1%29%7D%7B%7Binput%5C_high%7D%27-%7Binput%5C_low%7D%27%7D%20%5Cright%20%5Crceil)

![{input\_high}''=\frac{ZP-levels+1}{ZP}*{input\_low}'](https://latex.codecogs.com/gif.latex?%7Binput%5C_high%7D%27%27%3D%5Cfrac%7BZP-levels+1%7D%7BZP%7D*%7Binput%5C_low%7D%27)

![{input\_low}''=\frac{ZP}{ZP-levels+1}*{input\_high}'](https://latex.codecogs.com/gif.latex?%7Binput%5C_low%7D%27%27%3D%5Cfrac%7BZP%7D%7BZP-levels+1%7D*%7Binput%5C_high%7D%27)

![{input\_low,input\_high} = \begin{cases} {input\_low}',{input\_high}', & ZP \in $\{0,levels-1\}$ \\ {input\_low}',{input\_high}'', & {input\_high}'' - {input\_low}' > {input\_high}' - {input\_low}'' \\ {input\_low}'',{input\_high}', & {input\_high}'' - {input\_low}' <= {input\_high}' - {input\_low}''\\ \end{cases}](https://latex.codecogs.com/gif.latex?%7Binput%5C_low%2Cinput%5C_high%7D%20%3D%20%5Cbegin%7Bcases%7D%20%7Binput%5C_low%7D%27%2C%7Binput%5C_high%7D%27%2C%20%26%20ZP%20%5Cin%20%24%5C%7B0%2Clevels-1%5C%7D%24%20%5C%5C%20%7Binput%5C_low%7D%27%2C%7Binput%5C_high%7D%27%27%2C%20%26%20%7Binput%5C_high%7D%27%27%20-%20%7Binput%5C_low%7D%27%20%3E%20%7Binput%5C_high%7D%27%20-%20%7Binput%5C_low%7D%27%27%20%5C%5C%20%7Binput%5C_low%7D%27%27%2C%7Binput%5C_high%7D%27%2C%20%26%20%7Binput%5C_high%7D%27%27%20-%20%7Binput%5C_low%7D%27%20%3C%3D%20%7Binput%5C_high%7D%27%20-%20%7Binput%5C_low%7D%27%27%5C%5C%20%5Cend%7Bcases%7D)

You can use the `num_init_steps` parameter from the `initializer` group to initialize the values of `input_low` and `input_range` from the collected statistics during given number of steps.

#### Quantization Implementation

In our implementation, we use a slightly transformed formula. It is equivalent by order of floating-point operations to simplified symmetric formula and the assymetric one. The small difference is addition of small positive number `eps` to prevent division by zero and taking absolute value of range, since it might become negative on backward:

![output = \frac{\left\lfloor clamp(input-input\_low^{*}, level\_low, level\_high)*s\right \rceil} {s} + input\_low^{*}](https://latex.codecogs.com/gif.latex?output%20%3D%20%5Cfrac%7B%5Cleft%5Clfloor%20clamp%28input-input%5C_low%5E%7B*%7D%2C%20level%5C_low%2C%20level%5C_high%29*s%5Cright%20%5Crceil%7D%20%7Bs%7D%20&plus;%20input%5C_low%5E%7B*%7D)

![s = \frac{level\_high}{|input\_range^{*}| + eps}](https://latex.codecogs.com/gif.latex?s%20%3D%20%5Cfrac%7Blevel%5C_high%7D%7B%7Cinput%5C_range%5E%7B*%7D%7C%20&plus;%20eps%7D)

For asymmetric:  
![\\input\_low^{*} = input\_low \\ input\_range^{*} = input\_range ](https://latex.codecogs.com/gif.latex?%5C%5Cinput%5C_low%5E%7B*%7D%20%3D%20input%5C_low%20%5C%5C%20input%5C_range%5E%7B*%7D%20%3D%20input%5C_range)

For symmetric:  
![\\input\_low^{*} = 0 \\ input\_range^{*} = scale](https://latex.codecogs.com/gif.latex?%5C%5Cinput%5C_low%5E%7B*%7D%20%3D%200%20%5C%5C%20input%5C_range%5E%7B*%7D%20%3D%20scale)


**Algorithm Parameters**:
- `algorithm` - the name of the compression algorithm should be `quantization` in this case.
- `quantize_inputs` - if True, quantize input of the model (optional, default is True).
- `quantize_outputs` - if True, quantize outputs of the model (optional, default is False).
- `ignored_scopes` - blacklist: layers that should be excluded from the compression (optional).
- `target_scopes` - whitelist: only the layers listed here should be compressed (optional)
- `activations` - quantization parameters of activations 
    -  `bits` - number of bits,  `8` by default.
    -  `mode` - a type of the quantization.  Can be `symmetric` and `asymmetric`. The default value is `symmetric`.
    -  `signed` - if True, signed quantization is allowed in the model. `false` for activations and `true` for weights by default.
    -  `signed_scope` - list of layers that require signed quantization. (optional).
- `weights` - quantization parameters of weights
    -  `bits` - number of bits,  `8` by default.
    -  `mode` - a type of the quantization.  Can be one of these values: `symmetric` and `asymmetric`. `symmetric` by default.
- `initializer` - parameters of initialization stage.
    - `num_init_steps` - a number of steps to calculate per-layer activations statistics that can be used for quantization range initialization. `1` by default.


### Binarization

NNCF supports binarizing weights and activations for 2D convolutional PyTorch\* layers (Conv2D) *only*.

Weight binarization may be done in two ways, depending on the configuration file parameters - either via [XNOR binarization](https://arxiv.org/abs/1603.05279) or via [DoReFa binarization](https://arxiv.org/abs/1606.06160). For DoReFa binarization, the scale of binarized weights for each convolution operation is calculated as the mean of absolute values of non-binarized convolutional filter weights, while for XNOR binarization, each convolutional operation has scales that are calculated in the same manner, but _per input channel_ of the convolutional filter. Refer to the original papers for details.

Binarization of activations is implemented via binarizing inputs to the convolutional layers in the following way:

![\text{out} = s * H(\text{in} - s*t)](https://latex.codecogs.com/png.latex?%5Ctext%7Bout%7D%20%3D%20s%20*%20H%28%5Ctext%7Bin%7D%20-%20s*t%29)

In the formula above,   
 - ![\text{in}](https://latex.codecogs.com/png.latex?%5Ctext%7Bin%7D) - non-binarized activation values  
  - ![\text{out}](https://latex.codecogs.com/png.latex?%5Ctext%7Bout%7D) - binarized activation values  
 -  ![H(x)](https://latex.codecogs.com/png.latex?H%28x%29) is the Heaviside step function  
  - ![s](https://latex.codecogs.com/png.latex?s) and ![t](https://latex.codecogs.com/png.latex?t) are trainable parameters corresponding to binarization scale and threshold respectively

Training binarized networks requires special scheduling of the training process. For instance, binarizing a pretrained ResNet18 model on ImageNet is a four-stage process, with each stage taking a certain number of epochs. During the stage 1, the network is trained without any binarization. During the stage 2, the training continues with binarization enabled for activations only. During the stage 3, binarization is enabled both for activations and weights. Finally, during the stage 4 the optimizer learning rate, which was kept constant at previous stages, is decreased according to a polynomial law, while weight decay parameter of the optimizer is set to 0. The configuration files for the NNCF binarization algorithm allow to control certain parameters of this training schedule.

**Algorithm Parameters**:

- `algorithm` - the name of the compression algorithm. Set to `binarization` in this case
- `mode` - the mode of weight binarization. Set either to `xnor` (default) or `dorefa`
- `ignored_scopes` - blacklist: layers that should be excluded from binarization (optional)
- `target_scopes` - whitelist: only the layers listed here should be compressed (optional)
- `params` - parameters of the binarization algorithm:
   - `batch_multiplier` - allows to increase the effective batch size during training by accumulating the gradients each `batch_multiplier - 1` batches and only performing the optimization step during the next training batch. Increasing this may improve training quality, since binarized networks exhibit noisy gradients requiring larger batch sizes than could be accomodated by GPUs. Default is 1.
   - `activations_bin_start_epoch` - starting from this training epoch, the convolution activations become binarized. Default is 1.
   - `weights_bin_start_epoch` - starting from this training epoch, the convolution weights become binarized. Default is 1.
   - `lr_poly_drop_start_epoch` - (optional) starting from this training epoch, the learning rate of the optimizer drops to 0 according to a polynomial law.
   - `lr_poly_drop_duration_epochs` - specifies the duration of the learning rate polynomial drop stage in training epochs. Default is 30.
   - `disable_wd_start_epoch` - (optional) starting from this training epoch, the weight decay parameter of the optimizer is set to 0.


### Non-Structured Sparsity

Sparsity algorithm zeros weights in Convolutional and Fully-Connected layers in a non-structured way, 
so that zero values are randomly distributed inside the tensor. Most of the sparsity algorithms set less to zero the less important weights but the criteria of how they do it is different. The framework contains several implementations of sparsity methods.

#### RB-Sparsity

This section describes the Regularization-Based Sparsity (RB-Sparsity) algorithm implemented in this framework. The method is based on ![L_0](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cdpi%7B120%7D%20L_0)-regularization, with which parameters of the model tend to zero:

![||\theta||_0 = \sum_{i=0}^{|\theta|} \lbrack \theta_i = 0 \rbrack](https://latex.codecogs.com/png.latex?%5Cdpi%7B130%7D%20%7C%7C%5Ctheta%7C%7C_0%20%3D%20%5Csum_%7Bi%3D0%7D%5E%7B%7C%5Ctheta%7C%7D%20%5Clbrack%20%5Ctheta_i%20%3D%200%20%5Crbrack)

However, since the ![L_0](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cdpi%7B120%7D%20L_0)-norm is non-differentiable, we relax it by adding multiplicative noise to the model parameters:

![\theta_{sparse}^{(i)} = \theta_i \cdot \epsilon_i, \quad \epsilon_i \sim \mathcal{B}(p_i)](https://latex.codecogs.com/png.latex?%5Cdpi%7B130%7D%20%5Ctheta_%7Bsparse%7D%5E%7B%28i%29%7D%20%3D%20%5Ctheta_i%20%5Ccdot%20%5Cepsilon_i%2C%20%5Cquad%20%5Cepsilon_i%20%5Csim%20%5Cmathcal%7BB%7D%28p_i%29)

Here, ![\epsilon_i](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cdpi%7B120%7D%20%5Cepsilon_i) may be interpreted as a binary mask that selects which weights should be zeroed, hence we add the regularizing term to the objective function that encourages desired level of sparsity to our model:

![L_{sparse} = \mathbb{E}_{\epsilon \sim P_{\epsilon}} \lbrack \frac{\sum_{i=0}^{|\theta|} \epsilon_i}{|\theta|} - level \rbrack ^2](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20L_%7Bsparse%7D%20%3D%20%5Cmathbb%7BE%7D_%7B%5Cepsilon%20%5Csim%20P_%7B%5Cepsilon%7D%7D%20%5Clbrack%20%5Cfrac%7B%5Csum_%7Bi%3D0%7D%5E%7B%7C%5Ctheta%7C%7D%20%5Cepsilon_i%7D%7B%7C%5Ctheta%7C%7D%20-%20level%20%5Crbrack%20%5E2)

Since we can not directly optimize distribution parameters `p`, we store and optimize `p` in the logit form:

![s = \sigma^{-1}(p) = log (\frac{p}{1 - p})](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20s%20%3D%20%5Csigma%5E%7B-1%7D%28p%29%20%3D%20log%20%28%5Cfrac%7Bp%7D%7B1%20-%20p%7D%29)

and reparametrize sampling of ![\epsilon_i](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cdpi%7B120%7D%20%5Cepsilon_i)  as follows:

![\epsilon = \lbrack \sigma(s + \sigma^{-1}(\xi)) > \frac{1}{2} \rbrack, \quad \xi \sim \mathcal{U}(0,1)](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20%5Cepsilon%20%3D%20%5Clbrack%20%5Csigma%28s%20&plus;%20%5Csigma%5E%7B-1%7D%28%5Cxi%29%29%20%3E%20%5Cfrac%7B1%7D%7B2%7D%20%5Crbrack%2C%20%5Cquad%20%5Cxi%20%5Csim%20%5Cmathcal%7BU%7D%280%2C1%29)

With this reparametrization, probability of keeping a particular weight during the forward pass equals exactly to ![\mathbb{P}( \epsilon_i = 1) = p_i](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cmathbb%7BP%7D%28%20%5Cepsilon_i%20%3D%201%29%20%3D%20p_i). We only use weights with ![p_i > \frac{1}{2}](https://latex.codecogs.com/png.latex?%5Cinline%20p_i%20%3E%20%5Cfrac%7B1%7D%7B2%7D) at test time. To make the objective function differentiable, we treat threshold function ![t(x) = x > c](https://latex.codecogs.com/png.latex?%5Cinline%20t%28x%29%20%3D%20x%20%3E%20c) as straight through estimator i.e. ![\frac{d t}{dx} = 1](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cfrac%7Bd%20t%7D%7Bdx%7D%20%3D%201)

The method requires a long schedule of the training process in order to minimize the accuracy drop.

> **NOTE**: The known limitation of the method is that the sparsified CNN must include Batch Normalization layers which make the training process more stable.

**Algorithm parameters**:
- `algorithm` - the name of the compression algorithm is `rb_sparsity` in this case.
- `params` - parameters of the RB-sparsity algorithm:
   - `sparsity_training_steps` - an overall number of epochs that are used to train sparsity (usually it is different from the total number of training epochs). After this epoch, current sparsity masks are frozen, so only the model parameters are fine-tuned.
   - `schedule` - a type of scheduler that is used to increase the sparsity rate from `sparsity_init` to `sparsity_target`. Can be one of these values:
	   - `polynomial`, `exponential`, or `adaptive`, by default it is `exponential` for the RB algorithm and `polynomial` for the Magnitude one.
	   In this case, the following parameter should be defined in the same scope of parameters:
		   - `sparsity_init` - initial sparsity rate target (![1 - level](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%201%20-%20level) in the objective). For example, value `0.1` means that at the begging of training, the model is trained to have 10% of its weights zeroed.
		   - `sparsity_target` - sparsity rate target at the end of the schedule. For example, the value `0.5` means that at the step with the number of `sparsity_steps`, the model is trained to have 50% of its weights zeroed.
		   - `sparsity_steps` - the number of epochs during which the sparsity rate target is increased from 		`sparsity_init` to `sparsity_target` value. This parameter depends on the number of iterations in one epoch. For example, the parameter in sample configuration files is based on the fact that the training is run on the ImageNet dataset with batch size 256, which corresponds to about 4000 iterations in the epoch.
		   - `patience` - used in the case of `adaptive` scheduler only and defines the number of epochs for plateau.
		- `multistep` - this scheduler assumes that target sparsity rates and steps (in epochs) should be defined manually using `steps` and `sparsity_levels` lists.
- `ignored_scopes` - list of layers that should be excluded from the compression (optional).

> **NOTE**: In all our sparsity experiments, we used the Adam optimizer and initial learning rate `0.001` for model weights and sparsity mask.

#### Magnitude  Sparsity

The magnitude sparsity method implements a naive approach that is based on the assumption that the contribution of lower weights is lower so that they can be pruned. After each training epoch, the method calculates a threshold based on the current sparsity ratio and applies it to zero weights which are lower than this threshold using one of the options below:
- Weights are used as is during the threshold calculation procedure.
- Weights are normalized before the threshold calculation.

**Algorithm parameters**:
- `algorithm` - the name of the compression algorithm. In this case, it is `magnitude_sparsity`.

All other parameters are the same as in the RB-sparsity algorithm.

#### Constant  Sparsity

This type of sparsity provides the ability to preserve weights sparsity during the further stages of fine-tuning. This method assumes that we already trained the model using one of the sparsity methods from the NNCF.

**Algorithm parameters**:
- `algorithm` - the name of the compression algorithm. In this case, it is `const_sparsity`.
- `ignored_scopes` - list of layers that should be excluded from the compression (optional).
