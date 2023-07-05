# Common mmcv modules
from mmcls.datasets.builder import DATASETS, PIPELINES

# mmcls's model modules
from mmcls.models.builder import MODELS
from otx.v2.adapters.torch.mmcv.mmcls.modules import *
from otx.v2.adapters.torch.mmcv.registry import MMCVRegistry


class MMCLSRegistry(MMCVRegistry):
    def __init__(self, name="mmcls"):
        super().__init__(name)
        self.module_registry.update({"models": MODELS, "datasets": DATASETS, "pipelines": PIPELINES})


if __name__ == "__main__":
    registry = MMCLSRegistry()

    from torch import nn

    class NewEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

        def forward(self, x):
            return self.l1(x)

    class NewDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

        def forward(self, x):
            return self.l1(x)

    registry.register_module(name="A", module=NewEncoder)
    result_module = registry.get("A")
    print(result_module)
