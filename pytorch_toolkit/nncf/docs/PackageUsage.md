# Use Neural Network Compression Framework standalone
This is step by step tutorial that shows how to integrate NNCF package into the existing project. The use case implies that the user already has a training pipeline that reproduces training of the model in the floating point precision and pre-trained model. So the task is to compress this model in order to accelerate the inference time.
Here are the steps that should be done to use NNCF compression algorithm.
## Step 1: Install the NNCF package
- Clone the NNCF repository
- Install pre-requisites: `pip install -r requirements.txt`. We suggest creating and installing all the packages in the separate virtual environment.
- Enter the repository root folder and run the following command to  build the NNCF package:
 `python setup.py bdist_wheel`
 - Now you can find the package in the `dist/nncf-<vesion>.whl` and install it using `pip` tool.

## Step 2: Integrate compression methods
At this step you need to revise your training code and make several changes, which enable compression methods. Each compression method is implemented through three different entities: the algorithm itself, compression loss, and compression scheduler which controls the algorithm during the training process.

 - **Add** the following imports in the beginning of the training sample right after importing PyTorch:
 ```
from nncf.config import Config
from nncf.dynamic_graph import patch_torch_operators
from nncf.utils import print_statistics
from nncf.algo_selector import create_compression_algorithm
```
 - **Insert** the following call right after these imports:
	```
	patch_torch_operators()
	```
 - After you create an instance of your model and load pre-trained weights you can **create a compression algorithm** and wrap your model by making the following call:
	```
	compression_algo = create_compression_algorithm(model, config)
	model = compression_algo.model
	```
	Where `config` is a configuration file for compression methods where all the options and hyperparameters are specified. For more information about the configuration file please refer to the following [description](./Configuration.md).
 - Then you can **wrap your model** with `DataParallel` or `DistributedDataParallel` classes for multi-GPU training. In the case of distributed training you also need to call `compression_algo.distributed()` method.
 - In the **training loop** you should do the following changes:
	 - After inferencing the model take a compression loss and add it (using `+` operator) to the common loss. e.g. cross-entropy loss:
		```
		compression_loss = compression_algo.loss()
		loss = cross_entropy_loss + compression_loss
		```
	 - Call the scheduler `step()` after each training iteration:
		 ```
		  compression_algo.scheduler.step()
		 ```
	 - Call the scheduler `epoch_step()` after each training epoch
		  ```
		  compression_algo.scheduler.epoch_step()
		 ```

Note: You can find these changes in the samples published in the NNCF repository.
