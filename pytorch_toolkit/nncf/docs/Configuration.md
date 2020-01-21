# Configuration File Description

The Neural Network Compression Framework (NNCF) and the training samples are designed to work with the configuration file where hyperparameters of training and compression algorithm are specified. These parameters are organized as a dictionary and stored in a JSON file that is deserialized when the training starts. The JSON file allows using comments that are supported by the [jstyleson](https://github.com/linjackson78/jstyleson) Python package.
Logically all parameters are divided into two groups:
- **Training parameters** that are related to the training process and used by a training sample (for example, *learning rate*, *weight decay*, *optimizer type*, and others)
- **Compression parameters** that are related to the compression method (for example, *compression algorithm name*, *target sparsity ratio*, and others)

> **NOTE**: Compression parameters are optional and can be skipped. In this case, a regular model training is executed without any compression method.

## Example of a Configuration File

This is an example of the configuration file that contains training and compression parameters for two algorithms: sparsity and INT8 quantization which are applied to the MobileNet v2 model on the ImageNet classification task.  See the description of each setting as comments inside the file.

```yaml
{
    /* Training hyperparameters */
    "model": "mobilenetv2", // Model name
    "input_sample_size": [1, 3, 224, 224], // Input size of the model (including a "batch dimension")
    "num_classes": 1000, // Number of classes in the dataset
    "batch_size" : 256, // Batch size
    "checkpoint_save_dir": "results/snapshots", // A path where to dump best and last model snapshots, this is a log directory by default.
    "weights": "<SNAPSHOT_FILE_PATH>", // FP32 snapshot that is used as a starting point during the model compression

    // Optimizer parameters
    // Note that we used "Adam" optimizer in all our experiments due to its better convergence and stability.
    // In all our sparsity experiments we used initial learning rate "0.001" for model weights and sparsity mask.
    "optimizer": {
        "type": "Adam",
        "base_lr": 0.001,
        "schedule_type": "multistep",
        "steps": [20, 30, 40]
    },

    /* Compression hyperparameters */
    "compression": [
        {
            "algorithm": "rb_sparsity", // Compression algorithm name
            "params": {
                // A type of scheduler that is used to increase the sparsity rate from `sparsity_init` to `sparsity_target`.
                // By default it is `exponential` for the RB algorithm and `polynomial` for Magnitude one.
                "schedule": "exponential",
                "sparsity_init": 0.01, // Initial sparsity ratio, for example value "0.1" means that the method sets 10% of zero weights as a target after the training process starts
                "sparsity_target": 0.52, // Desired sparsity ratio, for example value "0.5" means that the method schedules the training to get 50% of zero weights in the end
                "sparsity_steps": 5, // Number of epochs at which sparsity ratio is increased from "sparsity_init" value up to "sparsity_target" value
                "sparsity_training_steps": 10 // Overall number of epochs that are used to train sparsity mask
            },
            "ignored_scopes": ["MobileNetV2/Sequential[features]/Sequential[0]/Conv2d[0]"] // Layers or blocks that are excluded from compression
        },
        {
            "algorithm": "quantization", // Compression algorithm name
            "initializer": {
                "num_init_steps": 1, // Number of steps to calculate per-layer activations statistics that can be used for "scale" initialization.
                "type": "min_max"  // Type of collected statistics. Value "min_max" means that scale is initialized by maximum value and sign of minimum value defines sign of activations.
            }
        }
    ]
}
```