MMPRETRAIN_AVAILABLE = True

try:
    import mmpretrain  # noqa: F401
except ImportError:
    MMPRETRAIN_AVAILABLE = False

from .dataset import Dataset
from .engine import MMPTEngine as Engine
from .model import build_model_from_config

__all__ = ["build_model_from_config", "Dataset", "Engine", "Registry"]
