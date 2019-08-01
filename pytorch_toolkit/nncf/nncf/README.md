
# Neural Network Compression Framework  (NNCF)

This is a PyTorch-based framework for neural networks compression.

## Key features:
- Support of quantization, sparsity, and quantization with sparsity algorithms with fine-tuning. Each setup requires only one additional stage of fine-tuning.
- Automatic model graph transformation. The model is wrapped by the custom class and additional layers are inserted in the PyTorch dynamic graph. The transformations are configurable.
- Common interface for compression methods.
- GPU-accelerated layers for fast model fine-tuning.
- Distributed training support.
- Export models to ONNX format that is supported by [OpenVINO](https://github.com/opencv/dldt) Toolkit.

## Compression algorithm architecture overview
The compression method is divided into three logical parts:
- The algorithm itself that implements API of `CompressionAlgorithm` class and controls compression algorithm logic.
- The scheduler that implements API of `CompressionScheduler` class and compression method scheduling control logic.
- The loss that implements API of `CompressionLoss` class and represents auxiliary loss function which is required by the compression method. It must be added to the main objective function responsible for accuracy metric and trained jointly.

Note: In general, the compression method may not have its own scheduler and loss then the default implementations are used in this case.

For more details please see [Algorithm API](../docs/Algorithm.md) description manual.

## Implemented compression methods
Each compression method receives its own hyperparameters that are organized as a dictionary and basically stored in JSON file that is deserialized when the training starts. Compression methods can be applied separately or together producing sparse, quantized or sparse + quantized models. For more information about the configuration files please refer to the samples.

### Uniform quantization  w/ fine-tuning
A uniform "fake" quantization method supports an arbitrary number of bits (>=2) which is used to represent weights and activations.
####  Symmetric quantization
The method performs differentiable sampling of the continuous signal (e.g. activations or weights) during forward pass, simulating inference with integer arithmetic. During the training we optimize the **scale** parameter that represents range (determined by min=-scale and max=scale) of the original signal using gradient descent.
The sampling formula is the following:
```math
output = round(clamp(input * \frac{level\_high}{scale}, level\_low, level\_high)) * \frac{scale}{level\_high}
```

Where `level_low` and `level_high` represent the range of the discrete signal.
 - For weights: $`level\_low=-2^{bits-1}+1, level\_high=2^{bits-1}-1`$
 - For unsigned activations: $`level\_low=0,\; level\_high=2^{bits}-1`$
 - For signed activations: $`level\_low=-2^{bits-1},\; level\_high=2^{bits}-1`$
 
 You can use `num_init_steps` parameter from `initializer` group to initialize the values of `scale` and determine what activation should be signed or unsigned from the collected statistics during this number of steps. 

**Algorithm parameters**:
- `algorithm` - the name of the compression algorithm should be `quantization` in this case.
- `bits` - number of bits,  `8` by default.
- `num_init_steps` - a number of steps to calculate per-layer activations statistics that can be used for **scale** initialization.
- `signed_activations` - if true then signed activations are allowed in the model (optional).
- `signed_activation_scopes` - list of layers that require singed activations (optional).
- `quantize_inputs` - if true than quantize input of the model (optional).
- `ignored_scopes` - list of layers that should be excluded from the compression (optional).

### Non-structured sparsity
Sparsity algorithm zeros weights in convolutional and fully-connected layers in a non-structured way, so that zero values are randomly distributed inside the tensor. Most of the sparsity algorithms set less to zero the less important weights but the criteria of how they do it is different. The framework contains several implementations of sparsity methods.

#### RB-Sparsity

This section describes the RB-Sparsity (Regularization-Based) algorithm that is implemented in this framework. The method is based on $`L_0`$ regularization that encourages parameters of the model to become zero:

```math
    ||\theta||_0 = \sum_{i=0}^{|\theta|} \lbrack \theta_i = 0 \rbrack
```

However, since the $`L_0`$ norm is non-differentiable, we relax it by adding multiplicative noise to the model parameters:

```math
    \theta_{sparse}^{(i)} = \theta_i \cdot \epsilon_i, \quad
    \epsilon_i \sim \mathcal{B}(p_i)
```

Here $`\epsilon_i`$ may be interpreted as a binary mask that selects which weights should be zeroed, hence we add the regularizing term to the objective function that encorages desired level of sparsity to our model:

```math
L_{sparse} = \mathbb{E}_{\epsilon \sim P_{\epsilon}} \lbrack \frac{\sum_{i=0}^{|\theta|} \epsilon_i}{|\theta|} - level \rbrack ^2
```

Since we can not directly optimize distribution parameters $`p`$, we store and optimize $`p`$ in logit form:

```math
s = \sigma^{-1}(p) = log (\frac{p}{1 - p})
```
and reparametrize sampling of $`\epsilon`$ as:

```math
    \epsilon = \lbrack \sigma(s + \sigma^{-1}(\xi)) > \frac{1}{2} \rbrack, \quad \xi \sim \mathcal{U}(0,1)
```
it can be shown that with this reparametrization, probablity of keeping a particular weight during the forward pass is exatcly $`\mathbb{P}( \epsilon_i = 1) = p_i`$. We only use weights with $`p_i > \frac{1}{2}`$ at test time. To make the objective function differentiable, we treat threshold function $`t(x) = [x > c]`$ as straight through estimator i.e. $`\frac{d t}{dx} = 1`$

The method requires a long schedule of the training process in order to minimize the accuracy drop.

**Note**: the known limitation of the method is that the sparsified CNN must include Batch Normalization layers which make the training process more stable.

**Algorithm parameters**:
- `algorithm` - the name of the compression algorithm should be `rb_sparsity` in this case.
- `params` - parameters of RB-sparsity algorithm:
   - `sparsity_training_steps` - an overall number of epochs that are used to train sparsity (in general case it is different from the total number of training epochs). After this epoch, current sparsity masks will be frozen, so only the model parameters will be fine-tuned.
   - `schedule` - a type of scheduler that is used to increase the sparsity rate from `sparsity_init` to `sparsity_target`. Can be one of these values:
	   - `polynomial`, `exponential`, or `adaptive`, by default it is `exponential` for RB algo and `polynomial` for Magnitude one.
	   In this case the following parameter should be defined in the same scope of parameters:
		   - `sparsity_init` - initial sparsity rate target ($`1 - level`$ in the objective), e.g. value `0.1` means that at the begging of training, the model will be trained to have 10% of its weights zeroed.
		   - `sparsity_target` - sparsity rate target at the end of the schedule, e.g. value `0.5` means that at the step with the number of `sparsity_steps` model will be trained to have 50% of its weights zeroed.
		   - `sparsity_steps` - the number of epochs during which the sparsity rate target will be increased from 		`sparsity_init` to `sparsity_target` value. This parameter depends on the number of iterations in one epoch. For example, the parameter in sample configuration files is based on the fact that the training is run on ImageNet dataset with batch size 256 which corresponds to ~4000 iterations in the epoch.
		   - `patience` - used in the case of `adaptive` scheduler only and defines the number of epochs for plateau.
		- `multistep` - this scheduler assumes that target sparsity rates and steps (in epochs) should be defined manually using `steps` and `sparsity_levels` lists.
- `ignored_scopes` - list of layers that should be excluded from the compression (optional).

**Note**: in all our sparsity experiments we used Adam optimizer and initial learning rate 0.001 for model weights and sparsity mask.

#### Magnitude  sparsity
The magnitude sparsity method implements a naive approach that is based on the assumption that the contribution of lower weights is lower so that they can be pruned. After each training epoch the method calculates a threshold based on the current sparsity ratio and uses it to zero weights which are lower than this threshold. And here there are two options:
- Weights are used as is during the threshold calculation procedure.
- Weights are normalized before threshold calculation.

**Algorithm parameters**:
- `algorithm` - the name of the compression algorithm should be `magnitude_sparsity` in this case.
- `update_mask_on_forward` - if true - mask is calculated on each forward call, otherwise - on selecting threshold only.   

All other parameters are equal to the RB-sparsity algorithm.

#### Constant  sparsity
This type of sparsity provides the ability to preserve weights sparsity during the further stages of fine-tuning. This method assumes that we already trained the model using one of the sparsity methods from NNCF framework.
**Algorithm parameters**:
- `algorithm` - the name of the compression algorithm should be `const_sparsity` in this case.
- `ignored_scopes` - list of layers that should be excluded from the compression (optional).
