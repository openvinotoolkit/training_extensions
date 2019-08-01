# Compression Method API

This is a description of the Compression Method API. The Neural Network Compression Framework (NNCF) is designed to be extendable and it is assumed that the end user can easily add new compression methods to the framework implementing an API which is specified [here](../common.py).
The compression method is divided into three logical parts:
- The algorithm itself that implements API of `CompressionAlgorithm` class and logic of compression method.
- The scheduler that implements API of `CompressionScheduler` class and compression method scheduling control logic.
- The loss that implements API of `CompressionLoss` class and loss function that is used in the compression method during the training process.

Note: In case a custom compression method does not implement its own scheduler and loss then the default implementations are used.
## `CompressionAlgorithm`
`CompressionAlgorithm` class is designed to represent the compression method and its logic. It should contain references to the `CompressionScheduler`, `CompressionLoss`, and compressing model instances that are used in the method so they are accessible in the training loop.
The class methods:
- `def __init__(self, model, config)` - the constructor of compression method class that takes two arguments:
	-  `model` - an instance of the model to be compressed
	- `config` - a dictionary that contains parameters of compression method
	- `input_size` - a tuple of values (B, C, H, W) specifying the input tensor
          dimensions for the model
- `def model(self)` - returns the compressing model.
- `def loss(self)` - returns the compression loss instance which is used in the method.
- `def scheduler(self)` - returns the scheduler instance which is used in the method.
- `def distributed(self)` - the method that is called when [distributed training](https://pytorch.org/tutorials/intermediate/dist_tuto.html) with multiple training processes is going to be used. It may be used to do some preparation inside the algorithm to support distributed training.
- `def statistics(self)`- returns a dictionary of printable statistics that are logged.
- `def initialize(self, data_loader)`- configures parameters of algorithm. Particularly, for quantization it calculates per-layer activations statistics on training dataset to choose proper range for quantization.
- `def export_model(self, filename)` - the method is called when the model is going to be exported for inference, e.g. to ONNX format. It is used to make some method-specific preparations of the model graph, like removing auxiliary layers that were used for the model compression, that are followed by the model exporting and dumping it to the output file.
	- `filename` - a path to the file where the exported model will be saved.

## `CompressionLoss`
`CompressionLoss` class is used to calculate an additional loss that is added to the base loss during the training process. It uses the model graph to measure variables and activations values of the layers during the loss construction. For example, the $L_0$-based sparsity algorithm calculates the number of non-zero weights in convolutional and fully-connected layers to construct the loss function.
The class methods:
- `def forward(self)` - the method is called to calculate the loss.
- `def statistics(self)` - the method returns a dictionary of printable statistics that are logged.

## `CompressionScheduler`
`CompressionScheduler` class is used to implement the logic of compression method control logic during the training process. It may change the method hyperparameters regarding what is the current training step or epoch. For example, the sparsity method can smoothly increase the sparsity rate over epochs.
The class methods:
- `def step(self, last=None)` - the method is called after each training iteration when the gradient descent step over the mini-batch has been performed.
	- `last` - specified step number.
- `def epoch_step(self, last=None)` - the method is called after the epoch finished.
	- `last` - specified epoch number.
