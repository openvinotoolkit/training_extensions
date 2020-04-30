"""
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from tqdm import tqdm
import torch

from examples.common.example_logger import logger
from examples.semantic_segmentation.utils.loss_funcs import do_model_specific_postprocessing


class Test:
    """Tests the ``model`` on the specified test dataset using the
    data loader, and loss criterion.

    Keyword arguments:
    - model (``nn.Module``): the model instance to test.
    - data_loader (``Dataloader``): Provides single or multi-process
    iterators over the dataset.
    - criterion (``Optimizer``): The loss criterion.
    - metric (```Metric``): An instance specifying the metric to return.
    - device (``torch.device``): An object representing the device on which
    tensors are allocated.
    - model_name: Name of the model to be trained - determines model-specific processing
    of the results (i.e. whether center crop should be applied, what outputs should be counted in metrics, etc.)

    """

    def __init__(self, model, data_loader, criterion, metric, device, model_name):
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.metric = metric
        self.device = device
        self.model_name = model_name

    def run_epoch(self, iteration_loss=False):
        """Runs an epoch of validation.

        Keyword arguments:
        - iteration_loss (``bool``, optional): Prints loss at every step.

        Returns:
        - The epoch loss (float), and the values of the specified metrics

        """
        self.model.eval()
        epoch_loss = 0.0
        self.metric.reset()
        for step, batch_data in tqdm(enumerate(self.data_loader), total=len(self.data_loader)):
            # Get the inputs and labels
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            with torch.no_grad():
                # Forward propagation
                outputs = self.model(inputs)

                labels, loss_outputs, metric_outputs = do_model_specific_postprocessing(self.model_name,
                                                                                        labels,
                                                                                        outputs)

                # Loss computation
                loss = self.criterion(loss_outputs, labels)

            # Keep track of loss for current epoch
            epoch_loss += loss.item()

            self.metric.add(metric_outputs.detach(), labels.detach())

            if iteration_loss:
                logger.info("[Step: {}] Iteration loss: {:.4f}".format(step, loss.item()))

        return epoch_loss / len(self.data_loader), self.metric.value()
