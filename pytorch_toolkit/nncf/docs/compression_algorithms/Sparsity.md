### Non-Structured Sparsity
Sparsity algorithm zeros weights in Convolutional and Fully-Connected layers in a non-structured way,
so that zero values are randomly distributed inside the tensor. Most of the sparsity algorithms set the less important weights to zero but the criteria of how they do it is different. The framework contains several implementations of sparsity methods.

#### RB-Sparsity

This section describes the Regularization-Based Sparsity (RB-Sparsity) algorithm implemented in this framework. The method is based on ![L_0](https://microsoft.codecogs.com/png.latex?%5Cinline%20%5Cdpi%7B120%7D%20L_0)-regularization, with which parameters of the model tend to zero:

![||\theta||_0 = \sum_{i=0}^{|\theta|} \lbrack \theta_i = 0 \rbrack](https://microsoft.codecogs.com/png.latex?%5Cdpi%7B130%7D%20%7C%7C%5Ctheta%7C%7C_0%20%3D%20%5Csum_%7Bi%3D0%7D%5E%7B%7C%5Ctheta%7C%7D%20%5Clbrack%20%5Ctheta_i%20%3D%200%20%5Crbrack)

However, since the ![L_0](https://microsoft.codecogs.com/png.latex?%5Cinline%20%5Cdpi%7B120%7D%20L_0)-norm is non-differentiable, we relax it by adding multiplicative noise to the model parameters:

![\theta_{sparse}^{(i)} = \theta_i \cdot \epsilon_i, \quad \epsilon_i \sim \mathcal{B}(p_i)](https://microsoft.codecogs.com/png.latex?%5Cdpi%7B130%7D%20%5Ctheta_%7Bsparse%7D%5E%7B%28i%29%7D%20%3D%20%5Ctheta_i%20%5Ccdot%20%5Cepsilon_i%2C%20%5Cquad%20%5Cepsilon_i%20%5Csim%20%5Cmathcal%7BB%7D%28p_i%29)

Here, ![\epsilon_i](https://microsoft.codecogs.com/png.latex?%5Cinline%20%5Cdpi%7B120%7D%20%5Cepsilon_i) may be interpreted as a binary mask that selects which weights should be zeroed, hence we add the regularizing term to the objective function that encourages desired level of sparsity to our model:

![L_{sparse} = \mathbb{E}_{\epsilon \sim P_{\epsilon}} \lbrack \frac{\sum_{i=0}^{|\theta|} \epsilon_i}{|\theta|} - level \rbrack ^2](https://microsoft.codecogs.com/png.latex?%5Cdpi%7B120%7D%20L_%7Bsparse%7D%20%3D%20%5Cmathbb%7BE%7D_%7B%5Cepsilon%20%5Csim%20P_%7B%5Cepsilon%7D%7D%20%5Clbrack%20%5Cfrac%7B%5Csum_%7Bi%3D0%7D%5E%7B%7C%5Ctheta%7C%7D%20%5Cepsilon_i%7D%7B%7C%5Ctheta%7C%7D%20-%20level%20%5Crbrack%20%5E2)

Since we can not directly optimize distribution parameters `p`, we store and optimize `p` in the logit form:

![s = \sigma^{-1}(p) = log (\frac{p}{1 - p})](https://microsoft.codecogs.com/png.latex?%5Cdpi%7B120%7D%20s%20%3D%20%5Csigma%5E%7B-1%7D%28p%29%20%3D%20log%20%28%5Cfrac%7Bp%7D%7B1%20-%20p%7D%29)

and reparametrize sampling of ![\epsilon_i](https://microsoft.codecogs.com/png.latex?%5Cinline%20%5Cdpi%7B120%7D%20%5Cepsilon_i)  as follows:

![\epsilon = \lbrack \sigma(s + \sigma^{-1}(\xi)) > \frac{1}{2} \rbrack, \quad \xi \sim \mathcal{U}(0,1)](https://microsoft.codecogs.com/png.latex?%5Cdpi%7B120%7D%20%5Cepsilon%20%3D%20%5Clbrack%20%5Csigma%28s%20&plus;%20%5Csigma%5E%7B-1%7D%28%5Cxi%29%29%20%3E%20%5Cfrac%7B1%7D%7B2%7D%20%5Crbrack%2C%20%5Cquad%20%5Cxi%20%5Csim%20%5Cmathcal%7BU%7D%280%2C1%29)

With this reparametrization, probability of keeping a particular weight during the forward pass equals exactly to ![\mathbb{P}( \epsilon_i = 1) = p_i](https://microsoft.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cmathbb%7BP%7D%28%20%5Cepsilon_i%20%3D%201%29%20%3D%20p_i). We only use weights with ![p_i > \frac{1}{2}](https://microsoft.codecogs.com/png.latex?%5Cinline%20p_i%20%3E%20%5Cfrac%7B1%7D%7B2%7D) at test time. To make the objective function differentiable, we treat threshold function ![t(x) = x > c](https://microsoft.codecogs.com/png.latex?%5Cinline%20t%28x%29%20%3D%20x%20%3E%20c) as straight through estimator i.e. ![\frac{d t}{dx} = 1](https://microsoft.codecogs.com/png.latex?%5Cinline%20%5Cfrac%7Bd%20t%7D%7Bdx%7D%20%3D%201)

The method requires a long schedule of the training process in order to minimize the accuracy drop.

> **NOTE**: The known limitation of the method is that the sparsified CNN must include Batch Normalization layers which make the training process more stable.

**RB sparsity configuration file parameters**:

```
{
    "algorithm": "rb_sparsity",
    "params": {
            "schedule": "multistep",  // The type of scheduling to use for adjusting the target sparsity level
            "patience": 3, // A regular patience parameter for the scheduler, as for any other standard scheduler. Specified in units of scheduler steps.
            "sparsity_init": 0.05,// "Initial value of the sparsity level applied to the model
            "sparsity_target": 0.7, // Target value of the sparsity level for the model
            "sparsity_steps": 3, // The default scheduler will do this many proportional target sparsity level adjustments, distributed evenly across 'sparsity_training_steps'.
            "sparsity_training_steps": 50, // The number of steps after which the sparsity mask will be frozen and no longer trained
            "multistep_steps": [10, 20], // A list of scheduler steps at which to transition to the next scheduled sparsity level (multistep scheduler only).
            "multistep_sparsity_levels": [0.2, 0.5] //Levels of sparsity to use at each step of the scheduler as specified in the 'multistep_steps' attribute. The firstsparsity level will be applied immediately, so the length of this list should be larger than the length of the 'steps' by one."
    },

    // A list of model control flow graph node scopes to be ignored for this operation - functions as a 'blacklist'. Optional.
    "ignored_scopes": []

    // A list of model control flow graph node scopes to be considered for this operation - functions as a 'whitelist'. Optional.
    // "target_scopes": []
}
```

> **NOTE**: In all our sparsity experiments, we used the Adam optimizer and initial learning rate `0.001` for model weights and sparsity mask.

#### Magnitude Sparsity

The magnitude sparsity method implements a naive approach that is based on the assumption that the contribution of lower weights is lower so that they can be pruned. After each training epoch the method calculates a threshold based on the current sparsity ratio and uses it to zero weights which are lower than this threshold. And here there are two options:
- Weights are used as is during the threshold calculation procedure.
- Weights are normalized before the threshold calculation.

**Magnitude sparsity configuration file parameters**:
```
{
    "algorithm": "magnitude_sparsity",
    "params": {
            "schedule": "multistep",  // The type of scheduling to use for adjusting the target sparsity level
            "patience": 3, // A regular patience parameter for the scheduler, as for any other standard scheduler. Specified in units of scheduler steps.
            "sparsity_init": 0.05,// "Initial value of the sparsity level applied to the model
            "sparsity_target": 0.7, // Target value of the sparsity level for the model
            "sparsity_steps": 3, // The default scheduler will do this many proportional target sparsity level adjustments, distributed evenly across 'sparsity_training_steps'.
            "sparsity_training_steps": 50, // The number of steps after which the sparsity mask will be frozen and no longer trained
            "multistep_steps": [10, 20], // A list of scheduler steps at which to transition to the next scheduled sparsity level (multistep scheduler only).
            "multistep_sparsity_levels": [0.2, 0.5] //Levels of sparsity to use at each step of the scheduler as specified in the 'multistep_steps' attribute. The firstsparsity level will be applied immediately, so the length of this list should be larger than the length of the 'steps' by one."
    },

    // A list of model control flow graph node scopes to be ignored for this operation - functions as a 'blacklist'. Optional.
    "ignored_scopes": []

    // A list of model control flow graph node scopes to be considered for this operation - functions as a 'whitelist'. Optional.
    // "target_scopes": []
}
```

#### Constant Sparsity
This special algorithm takes no additional parameters and is used when you want to load a checkpoint already trained with another sparsity algorithm and do other compression without changing the sparsity mask.

**Constant sparsity configuration file parameters**:
```
{
    "algorithm": "const_sparsity",
    // A list of model control flow graph node scopes to be ignored for this operation - functions as a 'blacklist'. Optional.
    "ignored_scopes": []

    // A list of model control flow graph node scopes to be considered for this operation - functions as a 'whitelist'. Optional.
    // "target_scopes": []
}).
```
