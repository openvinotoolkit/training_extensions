
### Uniform Quantization with Fine-Tuning

A uniform "fake" quantization method supports an arbitrary number of bits (>=2) which is used to represent weights and activations.
The method performs differentiable sampling of the continuous signal (for example, activations or weights) during forward pass, simulating inference with integer arithmetic.

#### Common Quantization Formula

Quantization is parametrized by clamping range and number of quantization levels. The sampling formula is the following:

![output = \frac{\left\lfloor (clamp(input; input\_low, input\_high)-input\_low)  *s\right \rceil}{s} + input\_low\\](https://microsoft.codecogs.com/gif.latex?output%20%3D%20%5Cfrac%7B%5Cleft%5Clfloor%20%28clamp%28input%3B%20input%5C_low%2C%20input%5C_high%29-input%5C_low%29%20*s%5Cright%20%5Crceil%7D%7Bs%7D%20&plus;%20input%5C_low%5C%5C)

![clamp(input; input\_low, input\_high) = min(max(input, input\_low), input\_high)))](https://microsoft.codecogs.com/gif.latex?clamp%28input%3B%20input%5C_low%2C%20input%5C_high%29%20%3D%20min%28max%28input%2C%20input%5C_low%29%2C%20input%5C_high%29%29%29)

![s=\frac{levels-1}{input\_high - input\_low}](https://microsoft.codecogs.com/gif.latex?s%3D%5Cfrac%7Blevels-1%7D%7Binput%5C_high%20-%20input%5C_low%7D)

`input_low` and `input_high` represent the quantization range and ![\left\lfloor\cdot\right \rceil](https://microsoft.codecogs.com/gif.latex?%5Cleft%5Clfloor%5Ccdot%5Cright%20%5Crceil) denotes rounding to the nearest integer.

####  Symmetric Quantization

During the training, we optimize the **scale** parameter that represents the range `[input_low, input_range]` of the original signal using gradient descent:

![input\_low=scale*\frac{level\_low}{level\_high}](https://microsoft.codecogs.com/gif.latex?input%5C_low%3Dscale*%5Cfrac%7Blevel%5C_low%7D%7Blevel%5C_high%7D)

![input\_high=scale](https://microsoft.codecogs.com/gif.latex?input%5C_high%3Dscale)

In the formula above, `level_low` and `level_high` represent the range of the discrete signal.
 - For weights:

    ![level\_low=-2^{bits-1}+1](https://microsoft.codecogs.com/png.latex?%5Cinline%20%5Cdpi%7B120%7D%20level%5C_low%3D-2%5E%7Bbits-1%7D&plus;1),

    ![level\_high=2^{bits-1}-1](https://microsoft.codecogs.com/png.latex?%5Cinline%20%5Cdpi%7B120%7D%20level%5C_high%3D2%5E%7Bbits-1%7D-1)

    ![levels=255](https://microsoft.codecogs.com/gif.latex?levels%3D255)

 - For unsigned activations:

    ![level\_low=0](https://microsoft.codecogs.com/png.latex?%5Cinline%20%5Cdpi%7B120%7D%20level%5C_low%3D0)

    ![level\_high=2^{bits}-1](https://microsoft.codecogs.com/png.latex?%5Cinline%20%5Cdpi%7B120%7D%20level%5C_high%3D2%5E%7Bbits%7D-1)

    ![levels=256](https://microsoft.codecogs.com/gif.latex?levels%3D256)

 - For signed activations:

    ![level\_low=-2^{bits-1}](https://microsoft.codecogs.com/png.latex?%5Cdpi%7B120%7D%20level%5C_low%3D-2%5E%7Bbits-1%7D)

    ![level\_high=2^{bits-1}-1](https://microsoft.codecogs.com/png.latex?%5Cinline%20%5Cdpi%7B120%7D%20level%5C_high%3D2%5E%7Bbits-1%7D-1)

    ![levels=256](https://microsoft.codecogs.com/gif.latex?levels%3D256)

For all the cases listed above, the common quantization formula is simplified after substitution of `input_low`, `input_high` and `levels`:

![output = \left\lfloor clamp(input * \frac{level\_high}{scale}, level\_low, level\_high)\right \rceil * \frac{scale}{level\_high}](https://microsoft.codecogs.com/gif.latex?output%20%3D%20%5Cleft%5Clfloor%20clamp%28input%20*%20%5Cfrac%7Blevel%5C_high%7D%7Bscale%7D%2C%20level%5C_low%2C%20level%5C_high%29%5Cright%20%5Crceil%20*%20%5Cfrac%7Bscale%7D%7Blevel%5C_high%7D)

Use the `num_init_steps` parameter from the `initializer` group to initialize the values of `scale` and determine which activation should be signed or unsigned from the collected statistics during given number of steps.

####  Asymmetric Quantization

During the training we optimize the **input_low** and **input_range** parameters using gradient descent:

![input\_high=input\_low + input\_range](https://microsoft.codecogs.com/gif.latex?input%5C_high%3Dinput%5C_low%20&plus;%20input%5C_range)

![levels=256](https://microsoft.codecogs.com/gif.latex?levels%3D256)

![level\_low=0](https://microsoft.codecogs.com/png.latex?%5Cinline%20%5Cdpi%7B120%7D%20level%5C_low%3D0)

![level\_high=2^{bits}-1](https://microsoft.codecogs.com/png.latex?%5Cinline%20%5Cdpi%7B120%7D%20level%5C_high%3D2%5E%7Bbits%7D-1)

For better accuracy, floating-point zero should be within quantization range and strictly mapped into quant (without rounding). Therefore, the following scheme is applied to ranges of weights and activations before quantization:

![{input\_low}' = min(input\_low, 0)](https://microsoft.codecogs.com/gif.latex?%7Binput%5C_low%7D%27%20%3D%20min%28input%5C_low%2C%200%29)

![{input\_high}' = max(input\_high, 0)](https://microsoft.codecogs.com/gif.latex?%7Binput%5C_high%7D%27%20%3D%20max%28input%5C_high%2C%200%29)

![ZP= \left\lfloor \frac{-{input\_low}'*(levels-1)}{{input\_high}'-{input\_low}'} \right \rceil ](https://microsoft.codecogs.com/gif.latex?ZP%3D%20%5Cleft%5Clfloor%20%5Cfrac%7B-%7Binput%5C_low%7D%27*%28levels-1%29%7D%7B%7Binput%5C_high%7D%27-%7Binput%5C_low%7D%27%7D%20%5Cright%20%5Crceil)

![{input\_high}''=\frac{ZP-levels+1}{ZP}*{input\_low}'](https://microsoft.codecogs.com/gif.latex?%7Binput%5C_high%7D%27%27%3D%5Cfrac%7BZP-levels+1%7D%7BZP%7D*%7Binput%5C_low%7D%27)

![{input\_low}''=\frac{ZP}{ZP-levels+1}*{input\_high}'](https://microsoft.codecogs.com/gif.latex?%7Binput%5C_low%7D%27%27%3D%5Cfrac%7BZP%7D%7BZP-levels+1%7D*%7Binput%5C_high%7D%27)

![{input\_low,input\_high} = \begin{cases} {input\_low}',{input\_high}', & ZP \in $\{0,levels-1\}$ \\ {input\_low}',{input\_high}'', & {input\_high}'' - {input\_low}' > {input\_high}' - {input\_low}'' \\ {input\_low}'',{input\_high}', & {input\_high}'' - {input\_low}' <= {input\_high}' - {input\_low}''\\ \end{cases}](https://microsoft.codecogs.com/gif.latex?%7Binput%5C_low%2Cinput%5C_high%7D%20%3D%20%5Cbegin%7Bcases%7D%20%7Binput%5C_low%7D%27%2C%7Binput%5C_high%7D%27%2C%20%26%20ZP%20%5Cin%20%24%5C%7B0%2Clevels-1%5C%7D%24%20%5C%5C%20%7Binput%5C_low%7D%27%2C%7Binput%5C_high%7D%27%27%2C%20%26%20%7Binput%5C_high%7D%27%27%20-%20%7Binput%5C_low%7D%27%20%3E%20%7Binput%5C_high%7D%27%20-%20%7Binput%5C_low%7D%27%27%20%5C%5C%20%7Binput%5C_low%7D%27%27%2C%7Binput%5C_high%7D%27%2C%20%26%20%7Binput%5C_high%7D%27%27%20-%20%7Binput%5C_low%7D%27%20%3C%3D%20%7Binput%5C_high%7D%27%20-%20%7Binput%5C_low%7D%27%27%5C%5C%20%5Cend%7Bcases%7D)

You can use the `num_init_steps` parameter from the `initializer` group to initialize the values of `input_low` and `input_range` from the collected statistics during given number of steps.

#### Quantization Implementation

In our implementation, we use a slightly transformed formula. It is equivalent by order of floating-point operations to simplified symmetric formula and the assymetric one. The small difference is addition of small positive number `eps` to prevent division by zero and taking absolute value of range, since it might become negative on backward:

![output = \frac{\left\lfloor clamp(input-input\_low^{*}, level\_low, level\_high)*s\right \rceil} {s} + input\_low^{*}](https://microsoft.codecogs.com/gif.latex?output%20%3D%20%5Cfrac%7B%5Cleft%5Clfloor%20clamp%28input-input%5C_low%5E%7B*%7D%2C%20level%5C_low%2C%20level%5C_high%29*s%5Cright%20%5Crceil%7D%20%7Bs%7D%20&plus;%20input%5C_low%5E%7B*%7D)

![s = \frac{level\_high}{|input\_range^{*}| + eps}](https://microsoft.codecogs.com/gif.latex?s%20%3D%20%5Cfrac%7Blevel%5C_high%7D%7B%7Cinput%5C_range%5E%7B*%7D%7C%20&plus;%20eps%7D)

For asymmetric:
![\\input\_low^{*} = input\_low \\ input\_range^{*} = input\_range ](https://microsoft.codecogs.com/gif.latex?%5C%5Cinput%5C_low%5E%7B*%7D%20%3D%20input%5C_low%20%5C%5C%20input%5C_range%5E%7B*%7D%20%3D%20input%5C_range)

For symmetric:
![\\input\_low^{*} = 0 \\ input\_range^{*} = scale](https://microsoft.codecogs.com/gif.latex?%5C%5Cinput%5C_low%5E%7B*%7D%20%3D%200%20%5C%5C%20input%5C_range%5E%7B*%7D%20%3D%20scale)


#### Mixed Precision Quantization

Quantization to lower precisions (e.g. 6, 4, 2 bits) is an efficient way to accelerate inference of neural networks. Though
NNCF supports quantization with an arbitrary number of bits to represent weights and activations values, choosing
ultra low bitwidth could noticeably affect the model's accuracy.
A good trade-off between accuracy and performance is achieved by assigning different precisions to different layers.
NNCF utilizes the [HAWQ-v2](https://arxiv.org/pdf/1911.03852.pdf) method to automatically choose optimal mixed precision
configuration by taking into account the sensitivity of each layer, i.e. how much lower-bit quantization of each layer
decreases the accuracy of model. The most sensitive layers are kept at higher precision. The sensitivity of the i-th layer is
calculated by multiplying the average Hessian trace with the L2 norm of quantization perturbation:

![\overline{Tr}(H_{i}) * \left \| Q(W_{i}) - W_{i} \right \|^2_2](https://microsoft.codecogs.com/gif.download?%5Coverline%7BTr%7D%28H_%7Bi%7D%29%20*%20%5Cleft%20%5C%7C%20Q%28W_%7Bi%7D%29%20-%20W_%7Bi%7D%20%5Cright%20%5C%7C%5E2_2)

The sum of the sensitivities for each layer forms a metric that is used to determine the specific bit precision
configuration. The optimal configuration is found by calculating this metric for all possible bitwidth settings and
selecting the median one. To reduce exponential search the following restriction is used: layers with a small value of
average Hessian trace are quantized to lower bits, and vice versa.

The Hessian trace is estimated with the randomized [Hutchinson algorithm](https://www.researchgate.net/publication/220432178_Randomized_Algorithms_for_Estimating_the_Trace_of_an_Implicit_Symmetric_Positive_Semi-Definite_Matrix).
Given Rademacher distributed random vector v, the trace of symmetric matrix H is equal to the estimation of a quadratic form:

![Tr(H) = \mathbb{E}[v^T H v]](https://microsoft.codecogs.com/gif.download?Tr%28H%29%20%3D%20%5Cmathbb%7BE%7D%5Bv%5ET%20H%20v%5D)

The randomized algorithm solves the expectation by Monte Carlo using sampling of v from its distribution, evaluating
the quadratic term, and averaging:

![Tr(H) \approx \frac{1}{m}\sum_{i=1}^{m}[v_i^T H v_i]](https://microsoft.codecogs.com/gif.download?Tr%28H%29%20%5Capprox%20%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%5Bv_i%5ET%20H%20v_i%5D)

Evaluation of the quadratic term happens by computing ![Hv](https://microsoft.codecogs.com/gif.download?Hv) - the result
of multiplication of the Hessian matrix with a given random vector v, without the explicit formation of the Hessian operator.
For gradient of the loss with respect to the i-th block ![g_i](https://microsoft.codecogs.com/gif.download?g_i) and for
a random vector v, which is independent of ![W_i](https://microsoft.codecogs.com/gif.download?W_i), we have the equation:

![\frac{\partial(g_i^T v)}{\partial  W_i} = H_i v](https://microsoft.codecogs.com/gif.download?%5Cfrac%7B%5Cpartial%28g_i%5ET%20v%29%7D%7B%5Cpartial%20W_i%7D%20%3D%20H_i%20v)

where ![H_i](https://microsoft.codecogs.com/gif.download?H_i) is the Hessian matrix of loss with respect to
![W_i](https://microsoft.codecogs.com/gif.download?W_i). Hence ![Hv](https://microsoft.codecogs.com/gif.download?Hv) can be
computed by 2 backpropagation passes: first  - with respect to the loss and second - with respect to the product of the
gradients and a random vector.   

Automatic mixed precision selection can be enabled by specifying `"type": "hawq"` in `precision` group within
`initializer` section of the quantization algorithm. The manual mode is also available by explicitly setting the number
of bits per layer through `bitwidth_per_scope` parameter.

**Quantization configuration file parameters**:
```
{
    "algorithm": "quantization",
    "initializer": {
        "range": {
            "num_init_steps": 5, // Number of batches from the training dataset to consume as sample model inputs for purposes of setting initial minimum and maximum quantization ranges
            "type": "minmax" // Type of the initializer - determines which statistics gathered during initialization will be used to initialize the quantization ranges
        },
        "precision": {
            "type": "hawq", // Type of precision initialization - either "manual" or "hawq". With "manual", precisions are defined explicitly via "bitwidth_per_scope". With "hawq", these are determined automatically using the HAWQ algorithm.
            "bits": [4, 8], // A list of bitwidth to choose from when performing precision initialization.",
            "num_data_points": 200, // Number of data points to iteratively estimate Hessian trace, 200 by default.
            "iter_number": 200, // Maximum number of iterations of Hutchinson algorithm to estimate Hessian trace, 200 by default
            "tolerance": 1e-5, //  Minimum relative tolerance for stopping the Hutchinson algorithm. It's calculated  between mean average trace from previous iteration and current one. 1e-5 by default
            "bitwidth_per_scope": [ // Manual settings for the quantizer bitwidths. Scopes are used to identify the weight quantizers. The same number of bits is assigned to adjacent activation quantizers. By default bitwidth is taken from global quantization parameters from `weights` and `activations` sections above
                [
                    4,
                    "MobileNetV2/Sequential[features]/InvertedResidual[8]/Sequential[conv]/NNCFConv2d[0]/ModuleDict[pre_ops]/UpdateWeight[0]/AsymmetricQuantizer[op]"
                ], // A tuple of a bitwidth and a scope
                [
                    4,
                    "ModuleDict/AsymmetricQuantizer[MobileNetV2/Sequential[features]/InvertedResidual[15]/Sequential[conv]/ReLU6[5]/hardtanh_0]"
                ]
            ]
        }
    }
    "weights": { // Constraints to be applied to model weights quantization only.Overrides higher-level settings.
        "mode": "symmetric", // Mode of quantization
        "bits": 8, // Bitwidth to quantize to.
        "signed": true, // Whether to use signed or unsigned input/output values for quantization. If specified as unsigned and the input values during initialization have differing signs, will reset to performing signed quantization instead.
        "per_channel": false, // Whether to quantize inputs per channel (i.e. per 0-th dimension for weight quantization,and per 1-st dimension for activation quantization)

        // A list of model control flow graph node scopes to be ignored for this operation - functions as a 'blacklist'. Optional.
        "ignored_scopes": []

        // A list of model control flow graph node scopes to be considered for this operation - functions as a 'whitelist'. Optional.
        // "target_scopes": []
    },
    "activations": { // Constraints to be applied to model activations quantization only. Overrides higher-level settings.
        "mode": "symmetric", // Mode of quantization
        "bits": 4, // Bitwidth to quantize to.
        "signed": true, // Whether to use signed or unsigned input/output values for quantization. If specified as unsigned and the input values during initialization have differing signs, will reset to performing signed quantization instead.
        "per_channel": false, // Whether to quantize inputs per channel (i.e. per 0-th dimension for weight quantization,and per 1-st dimension for activation quantization)

        // A list of model control flow graph node scopes to be ignored for this operation - functions as a 'blacklist'. Optional.
        "ignored_scopes": []

        // A list of model control flow graph node scopes to be considered for this operation - functions as a 'whitelist'. Optional.
        // "target_scopes": []
    },
    "quantize_inputs": true, // Whether the model inputs should be immediately quantized prior to any other model operations."
    "quantizable_subgraph_patterns": [ // Each sub-list in this list will correspond to a sequence of operations in the model control flow graph that will have a quantizer appended at the end of the sequence
        [
            "cat",
            "batch_norm"
        ],
        [
            "h_swish"
        ]
    ]
    "scope_overrides": { // This option is used to specify overriding quantization constraints for specific scope, e.g. in case you need to quantize a single operation differently than the rest of the model.
        "{re}.*InvertedResidual.*": {
            "mode": "symmetric", // Mode of quantization
            "bits": 4, // Bitwidth to quantize to.
            "signed": true, // Whether to use signed or unsigned input/output values for quantization. If specified as unsigned and the input values during initialization have differing signs, will reset to performing signed quantization instead.
            "per_channel": false // Whether to quantize inputs per channel (i.e. per 0-th dimension for weight quantization,and per 1-st dimension for activation quantization)
        }
    },

    // A list of model control flow graph node scopes to be ignored for this operation - functions as a 'blacklist'. Optional.
    "ignored_scopes": []

    // A list of model control flow graph node scopes to be considered for this operation - functions as a 'whitelist'. Optional.
    // "target_scopes": []
}
```
