### Pruning

#### Filter pruning    

 Filter pruning algorithm zeros output filters in Convolutional layers based on some filter importance criterion  (filters with smaller importance are pruned).     
The framework contains three such criteria: `L1`, `L2` norm, and `Geometric Median`. Also, different schemes of pruning application are presented by different schedulers.    

#### Filter importance criterions **L1, L2**

 `L1`, `L2` filter importance criteria are based on the following assumption:    
> Convolutional filters with small ![l_p](https://microsoft.codecogs.com/gif.latex?l_p) norms do not significantly contribute to output activation values, and thus have a small impact on the final predictions of CNN models.   
In the above, the ![l_p](https://microsoft.codecogs.com/gif.latex?l_p) norm for filter F is:    

![||F||_p = \sqrt[p]{\sum_{c, k_1, k_2 = 1}^{C, K, K}|F(c, k_1, k_2)|^p}](https://microsoft.codecogs.com/gif.latex?%7C%7CF%7C%7C_p%20%3D%20%5Csqrt%5Bp%5D%7B%5Csum_%7Bc%2C%20k_1%2C%20k_2%20%3D%201%7D%5E%7BC%2C%20K%2C%20K%7D%7CF%28c%2C%20k_1%2C%20k_2%29%7C%5Ep%7D)  

During the pruning procedure filters with smaller  `L1` or `L2` norm will be pruned first.    

**Geometric Median**

Usage of the geometric median filter importance criterion is based on the following assumptions:  
> Let ![\{F_i, \dots , F_j\}](https://microsoft.codecogs.com/gif.latex?%5C%7BF_i%2C%20..%20%2C%20F_j%5C%7D) be the set of N filters in a convolutional layer that are closest to the geometric median of all the filters in that layer.   As it was shown, each of those filters can be decomposed into a linear combination of the rest of the filters further from the geometric median with a small error. Hence, these filters can be pruned without much impact on network accuracy.  Since we have only fixed number of filters in each layer and the task of calculation of geometric median is a non-trivial problem in computational geometry, we can instead find which filters minimize the summation of the distance with other filters.  

Then Geometric Median importance of ![F_i](https://microsoft.codecogs.com/gif.latex?F_i) filter from ![L_j](https://microsoft.codecogs.com/gif.latex?L_j) layer is:  
![G(F_i) = \sum_{F_j \in \{F_1, \dots F_m\}, j\neq i} ||F_i - F_j||_2](https://microsoft.codecogs.com/gif.latex?G%28F_i%29%20%3D%20%5Csum_%7BF_j%20%5Cin%20%5C%7BF_1%2C%20%5Cdots%20F_m%5C%7D%2C%20j%5Cneq%20i%7D%20%7C%7CF_i%20-%20F_j%7C%7C_2)  
Where ![L_j](https://microsoft.codecogs.com/gif.latex?L_j) is j-th convolutional layer in model. ![\{F_1, \dots F_m\} \in L_j](https://microsoft.codecogs.com/gif.latex?%5C%7BF_1%2C%20%5Cdots%20F_m%5C%7D%20%5Cin%20L_j) - set of all  output filters in ![L_j](https://microsoft.codecogs.com/gif.latex?L_j) layer.  

Then during pruning filters with smaller ![G(F_i)](https://microsoft.codecogs.com/gif.latex?G%28F_i%29) importance function will be pruned first.  

#### Schedulers

**Baseline Scheduler**    
 Firstly, during `num_init_steps` epochs the model is trained without pruning. Secondly, the pruning algorithm calculates filter importances and prune a `pruning_target` part of the filters with the smallest importance in each convolution.   
The zeroed filters are frozen afterwards and the remaining model parameters are fine-tuned.   

**Parameters of the scheduler:**  
 - `num_init_steps` - number of epochs for model pretraining **before** pruning.    
    - `pruning_target` - pruning rate target . For example, the value `0.5` means that right after pretraining, at the epoch    
     with the number of `num_init_steps`, the model will have 50% of its convolutional filters zeroed.   


**Exponential scheduler**   

Similar to the Baseline scheduler, during `num_init_steps` epochs model is pretrained without pruning.  
During the next `pruning steps` epochs `Exponential scheduler` gradually increasing pruning rate from `pruning_init` to `pruning_target`. After each pruning training epoch pruning algorithm calculates filter importances for all convolutional filters and prune (setting to zero) `current_pruning_rate` part of filters with the smallest importance in each Convolution.  After `num_init_steps` + `pruning_steps` epochs algorithm with zeroed filters is frozen and remaining model parameters only fine-tunes.    

Current pruning rate ![P_{i}](https://microsoft.codecogs.com/svg.latex?P_{i}) (on i-th epoch) during training calculates by equation:  
![P_i = a * e^{- k * i}](https://microsoft.codecogs.com/gif.latex?P_i%20%3D%20a%20*%20e%5E%7B-%20k%20*%20i%7D)  
Where ![a, k](https://microsoft.codecogs.com/svg.latex?a,%20k) - parameters.  

**Parameters of scheduler:**  
 - `num_init_steps` - number of epochs for model pretraining before pruning.    
    - `pruning_steps` - the number of epochs during which the pruning rate target is increased from  `pruning_init` to `pruning_target` value.    
    - `pruning_init` - initial pruning rate target. For example, value `0.1` means that at the begging of training, the model is trained to have 10% of convolutional filters zeroed.    
    - `pruning_target` - pruning rate target at the end of the schedule. For example, the value `0.5` means that at the epoch with the number of `num_init_steps + pruning_steps`, the model is trained to have 50% of its convolutional filters zeroed.    

**Exponential with bias scheduler**   
Similar to the `Exponential scheduler`, but current pruning rate ![P_{i}](https://microsoft.codecogs.com/svg.latex?P_{i}) (on i-th epoch) during training calculates by equation:  
![P_i = a * e^{- k * i} + b](https://microsoft.codecogs.com/gif.latex?P_i%20%3D%20a%20*%20e%5E%7B-%20k%20*%20i%7D%20&plus;%20b)  
Where ![a, k, b](https://microsoft.codecogs.com/gif.latex?a%2C%20k%2C%20b) - parameters.  

> **NOTE**:  Baseline scheduler prunes filters only ONCE and after it just fine-tunes remaining parameters while exponential (and exponential with bias) schedulers choose and prune different filters subsets at each pruning epoch.  

**Filter pruning configuration file parameters**:    
```{
    "algorithm": "filter_pruning",
    "params": {
        "schedule": "baseline", // The type of scheduling to use for adjusting the target pruning level. Either `exponential`, `exponential_with_bias`,  or `baseline`, by default it is `baseline`"
        "pruning_init": 0.1, // Initial value of the pruning level applied to the model. 0.0 by default.
        "pruning_target": 0.4, // Target value of the pruning level for the model. 0.5 by default.
        "num_init_steps": 3, // Number of epochs for model pretraining before starting filter pruning. 0 by default.
        "pruning_steps": 10, // Number of epochs during which the pruning rate is increased from `pruning_init` to `pruning_target` value.
        "weight_importance": "L2", // The type of filter importance metric. Can be one of `L1`, `L2`, `geometric_median`. `L2` by default.
        "all_weights": false, // Whether to prune layers independently (choose filters with the smallest importance in each layer separately) or not. `False` by default.
        "prune_first_conv": false, // Whether to prune first Convolutional layers or not. First means that it is a convolutional layer such that there is a path from model input to this layer such that there are no other convolution operations on it. `False` by default. 
        "prune_last_conv": false, // Whether to prune last Convolutional layers or not.  Last means that it is a Convolutional layer such that there is a path from this layer to the model output such that there are no other convolution operations on it. `False` by default. 
        "prune_downsample_convs": false, // Whether to prune downsample Convolutional layers (with stride > 1) or not. `False` by default.
        "prune_batch_norms": false, // Whether to nullifies parameters of Batch Norm layer corresponds to zeroed filters of convolution corresponding to this Batch Norm. `False` by default.
        "zero_grad": true // Whether to setting gradients corresponding to zeroed filters to zero during training, `True` by default.    
    },

    // A list of model control flow graph node scopes to be ignored for this operation - functions as a 'blacklist'. Optional.
    "ignored_scopes": []

    // A list of model control flow graph node scopes to be considered for this operation - functions as a 'whitelist'. Optional.
    // "target_scopes": []
}```

> **NOTE:**  In all our pruning experiments we used SGD optimizer.
