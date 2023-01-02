from abc import ABC, abstractmethod

import torch.nn as nn


class BaseExplainer(ABC, nn.Module):
    """
    Blackbox explainer base class
    """

    def __init__(self, model):
        self._model = model

    @abstractmethod
    def get_saliency_map(self):
        pass

    @abstractmethod
    def run(self):
        pass
