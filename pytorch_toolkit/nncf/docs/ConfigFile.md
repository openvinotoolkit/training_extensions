# NNCF Configuration File Description

The Neural Network Compression Framework (NNCF) is designed to work with the configuration file where the parameters of compression that should be applied to the model are specified. 
These parameters are organized as a dictionary and stored in a JSON file that is deserialized when the training starts. 
The JSON file allows using comments that are supported by the [jstyleson](https://github.com/linjackson78/jstyleson) Python package.

Below is an example of the NNCF configuration file

```
{
    "input_info": [ // Required - describe the specifics of your model inputs here. This information is used to build the internal graph representation that is leveraged for proper compression functioning, and for exporting the compressed model to ONNX. Inputs in the array without a "keyword" attribute are described in the order of the model's "forward" function argument order.
                    {
                        "sample_size": [1, 3, 224, 224],  // Shape of the tensor expected as input to the model.
                        "type": "float", // Data type of the model input tensor.
                        "filler": "zeros", //  Determines what the tensor will be filled with when passed to the model    during tracing and exporting.
                        "keyword": "input" // Keyword to be used when passing the tensor to the model's 'forward' method. Optional.
                    },
                    // Provide a description for each model input.
                    {
                        "sample_size": [1, 1, 28, 28],
                        "type": "long",
                        "filler": "ones",
                        "keyword": "another_input"
                    },

                },
            ],
        },
        "hw_config_type": "cpu", // If specified, the compression algorithms will use parameter presets that are more likely to result in best performance on a given HW type. Optional.
        "compression": [ // One or more definitions for the compression algorithms to be applied to the model; either a single JSON object or an array of JSON objects. See README for each compression algorithm for a description of the available config parameters.
            {
                "algorithm": quantization,
                "initializer": {
                    "range": {
                        "num_init_steps": 10
                    }
                }
            },
            {
                "algorithm": "magnitude_sparsity",
                "params": {
                    "schedule": "multistep",
                    "steps": [
                        5,
                        10,
                        20,
                        30,
                        40
                    ],
                    "sparsity_levels": [
                        0.1,
                        0.2,
                        0.3,
                        0.4,
                        0.5,
                        0.6
                    ]
                }
            }
        ]
    }
}
```


The "compression" section is the core of the configuration file.
It defines the specific compression algorithms that are to be applied to the model.
You can specify either a single compression algorithm to be applied to the model, or multiple compression algorithms to be applied at once.
To specify a single compression algorithm, the "compression" section should hold a dictionary corresponding to one of the built-in NNCF compression algorithms.
The dictionary should contain the name of the desired algorithm and the key-value pairs of the configurable parameters of the algorithm.
See [Algorithms.md](./Algorithms.md) for a list of supported algorithms and their configuration formats.
To specify multiple compression algorithm at once, the "compression" section should hold an array (list) of dictionaries each corresponding to a single compression algorithm.

**IMPORTANT:** The `"ignored_scopes"` and `"target_scopes"` sections use a special string format (see "Operation addressing and scope" in [NNCFArchitecture.md](./NNCFArchitecture.md)) to specify the parts of the model that the compression should be applied to. For all such section, regular expression matching can be enabled by prefixing the string with `{re}`, which helps to specify the same compression pattern concisely for networks with multiple same-structured blocks such as ResNet or BERT.

The [example scripts](../examples) use the same configuration file structure to specify compression, but extend it at the root level to specify the training pipeline hyperparameters as well.
These extensions are training pipeline-specific rather than NNCF-specific and their format differs across the example scripts.

> **NOTE**: You can extend the configuration file format for custom usage in your own training pipeline, but only at the root level of the JSON file and only for the keywords that are not already used by NNCF.
If you wish to extend NNCF with a new compression algorithm (which is to be specified in the "compression" section), you should register it within the NNCF config [JSON schema](../nncf/config_schema.py), against which the config files are validated.
