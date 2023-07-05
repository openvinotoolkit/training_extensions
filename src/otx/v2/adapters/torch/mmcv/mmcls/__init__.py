MMCLS_AVAILABLE = True

try:
    import mmcls  # noqa: F401
except ImportError:
    MMCLS_AVAILABLE = False


from .dataset import Dataset
from .engine import MMCLSEngine as Engine
from .model import build_model_from_config

__all__ = ["build_model_from_config", "Dataset", "Engine"]
