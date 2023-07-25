# Common mmpretrain modules
from mmpretrain.datasets import *
from mmpretrain.engine import *
from mmpretrain.evaluation import *
from mmpretrain.models import *
from mmpretrain.registry import (
    BATCH_AUGMENTS,
    DATA_SAMPLERS,
    DATASETS,
    EVALUATORS,
    HOOKS,
    LOG_PROCESSORS,
    LOOPS,
    METRICS,
    MODEL_WRAPPERS,
    MODELS,
    OPTIM_WRAPPER_CONSTRUCTORS,
    OPTIM_WRAPPERS,
    OPTIMIZERS,
    PARAM_SCHEDULERS,
    RUNNER_CONSTRUCTORS,
    RUNNERS,
    TASK_UTILS,
    TRANSFORMS,
    VISBACKENDS,
    VISUALIZERS,
    WEIGHT_INITIALIZERS,
)
from mmpretrain.structures import *
from mmpretrain.visualization import *
from otx.v2.adapters.torch.mmengine.mmpretrain.modules import *
from otx.v2.adapters.torch.mmengine.registry import MMEngineRegistry

REGISTRY_LIST = [
    RUNNERS,
    RUNNER_CONSTRUCTORS,
    LOOPS,
    HOOKS,
    LOG_PROCESSORS,
    OPTIMIZERS,
    OPTIM_WRAPPERS,
    OPTIM_WRAPPER_CONSTRUCTORS,
    PARAM_SCHEDULERS,
    DATASETS,
    DATA_SAMPLERS,
    TRANSFORMS,
    MODELS,
    MODEL_WRAPPERS,
    WEIGHT_INITIALIZERS,
    BATCH_AUGMENTS,
    TASK_UTILS,
    METRICS,
    EVALUATORS,
    VISUALIZERS,
    VISBACKENDS,
]


class MMPretrainRegistry(MMEngineRegistry):
    def __init__(self, name="mmpretrain"):
        super().__init__(name)
        self.registry_dict = {registry.name: registry for registry in REGISTRY_LIST}


if __name__ == "__main__":
    registry = MMPretrainRegistry()

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

    registry.register_module(type="model", name="A", module=NewEncoder)
    result_module = registry.get("A")
    print(registry)
