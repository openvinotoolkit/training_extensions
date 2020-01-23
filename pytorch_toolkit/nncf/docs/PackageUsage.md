# Use Neural Network Compression Framework (NNCF) as Standalone

This is a step-by-step tutorial on how to integrate the NNCF package into the existing project. The
use case implies that the user already has a training pipeline that reproduces training of the model
in the floating-point precision and pretrained model. The task is to compress this model in order to
accelerate the inference time.
Follow the steps below to use the NNCF compression algorithm.

## Step 1: Install the NNCF Package

- Clone the NNCF repository
- Install prerequisites: `pip install -r requirements.txt`. We suggest creating and installing all the packages in the separate virtual environment.
- Enter the repository root folder and run the following command to  build the NNCF package:
 `python setup.py bdist_wheel`
 - Now you can find the package in `dist/nncf-<vesion>.whl` and install it using the `pip` tool.

## Step 2: Integrate Compression Methods

At this step, you need to revise your training code and make several changes, which enable compression methods. Each compression method is implemented through three different entities: the algorithm itself, compression loss, and compression scheduler, which controls the algorithm during the training process.

 1. **Add** the following import commands in the beginning of the training sample right after importing PyTorch:
	```
	from nncf.config import Config
	from nncf.dynamic_graph import patch_torch_operators
	from nncf.utils import print_statistics
	from nncf.algo_selector import create_compression_algorithm
	```
 2. **Insert** the following call right after these import commands:
	```
	patch_torch_operators()
	```
 3. After you create an instance of your model and load pretrained weights, **create a compression algorithm** and wrap your model by making the following call:
	```
	compression_algo = create_compression_algorithm(model, config)
	model = compression_algo.model
	```
	The `config` is a configuration file for compression methods where all the options and hyperparameters are specified. For more information about the configuration file, refer to its [description](./Configuration.md).
 4. **Wrap your model** with `DataParallel` or `DistributedDataParallel` classes for multi-GPU training. In the case of distributed training, call the `compression_algo.distributed()` method as well.
 5. Call the `compression_algo.initialize()` method before the start of your training loop. Some compression algorithms like quantization require arguments (`train_loader` for your training dataset) to be supplied to the `initialize()` method.
 - In the **training loop**, do the following changes:
	 - After inferring the model, take a compression loss and add it (using the `+` operator) to the common loss, for example cross-entropy loss:
		```
		compression_loss = compression_algo.loss()
		loss = cross_entropy_loss + compression_loss
		```
	 - Call the scheduler `step()` after each training iteration:
		```
		compression_algo.scheduler.step()
		```
	 - Call the scheduler `epoch_step()` after each training epoch:
		```
		compression_algo.scheduler.epoch_step()
		```

> **NOTE**: Find these changes in the samples published in the NNCF repository.

Important points you should consider when training your networks with compression algorithms:
  - Turn off the `Dropout` layers (and similar ones like `DropConnect`) when training a network with quantization or sparsity
  - It is better to turn off additional regularization in the loss function (for example, L2 regularization via `weight_decay`) when training the network with RB sparsity, since it already imposes an L0 regularization term.
